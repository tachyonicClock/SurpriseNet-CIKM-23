import typing as t

import torch
import torch.nn as nn
from surprisenet.activation import (
    ActivationStrategy,
    NaiveSurpriseNetActivation,
)
from experiment.strategy import ForwardOutput
from hvae.hvaeoodd.oodd.layers.linear import NormedDense, NormedLinear
from network.deep_vae import FashionMNISTDeepVAE
from network.trait import (
    AutoEncoder,
    Classifier,
    ConditionedSample,
    Decoder,
    Encoder,
    InferTask,
    MultiOutputNetwork,
    ParameterMask,
    SurpriseNet,
    Samplable,
    VariationalAutoEncoder,
)
from surprisenet.mask import ModuleDecorator, WeightMask
from surprisenet.task_inference import TaskInferenceStrategy
from torch import Tensor
from torch.nn import functional as F


class _PnBatchNorm(ParameterMask, ModuleDecorator):
    """BatchNorm is insanely annoying"""

    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__(wrappee)
        self._z_top: torch.Tensor
        self.register_buffer("_frozen", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_z_top", torch.tensor(0, dtype=torch.int))
        """Index top of the 'stack'. Should only increase"""

    @property
    def frozen(self) -> bool:
        return self._frozen.item()

    @frozen.setter
    def frozen(self, frozen: bool):
        self._frozen.fill_(frozen)

    def prune(self, to_prune_proportion: float) -> None:
        # Hopefully this freezes batch norm
        self.wrappee.weight.requires_grad = False
        self.wrappee.bias.requires_grad = False
        self.frozen = True

    def forward(self, input: Tensor) -> Tensor:
        if self.frozen:
            # Keep it in eval mode once we have frozen things
            self.wrappee.eval()
        return self.wrappee.forward(input)

    def _zero_grad(self, grad: Tensor):
        grad.fill_(0)

    def push_pruned(self) -> None:
        self._z_top += 1

    def activate_subsets(self, subset_ids: t.List[int]):
        pass

    def activate_task_id(self, task_id: int):
        pass

    def mutable_activate_subsets(self, subset_ids: t.List[int]):
        pass

    def subset_count(self) -> int:
        assert False

    def unfreeze_all(self):
        self.wrappee.weight.requires_grad = True
        self.wrappee.bias.requires_grad = True
        self.frozen = False


class _PnLinear(WeightMask):
    def __init__(self, wrappee: nn.Linear) -> None:
        self.wrappee: nn.Linear
        super().__init__(wrappee)

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    @property
    def in_features(self) -> Tensor:
        return self.wrappee.in_features

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.available_weights(), self.bias)


class _PnConv2d(WeightMask):
    def __init__(self, wrappee: nn.Conv2d) -> None:
        wrappee: nn.Conv2d
        super().__init__(wrappee)

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    @property
    def transposed(self) -> bool:
        return self.wrappee.transposed

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee._conv_forward(input, self.available_weights(), self.bias)


class _PnConvTransposed2d(WeightMask):
    def __init__(self, wrappee: nn.ConvTranspose2d) -> None:
        wrappee: nn.ConvTranspose2d
        super().__init__(wrappee)

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    @property
    def transposed(self) -> bool:
        return self.wrappee.transposed

    def forward(
        self, input: Tensor, output_size: t.Optional[t.List[int]] = None
    ) -> Tensor:
        w = self.wrappee
        if w.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(w.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = w._output_padding(
            input, output_size, w.stride, w.padding, w.kernel_size, w.dilation
        )  # type: ignore[arg-type]

        return F.conv_transpose2d(
            input,
            self.available_weights(),
            w.bias,
            w.stride,
            w.padding,
            output_padding,
            w.groups,
            w.dilation,
        )


def wrap(wrappee: nn.Module):
    # Remove Weight Norm
    if hasattr(wrappee, "weight_g") and hasattr(wrappee, "weight_v"):
        torch.nn.utils.remove_weight_norm(wrappee)
    if isinstance(wrappee, (NormedLinear, NormedDense)):
        wrappee.weightnorm = False

    # Recursive cases
    if isinstance(wrappee, nn.Linear):
        return _PnLinear(wrappee)
    elif isinstance(wrappee, nn.Conv2d):
        return _PnConv2d(wrappee)
    elif isinstance(wrappee, nn.ConvTranspose2d):
        return _PnConvTransposed2d(wrappee)
    elif isinstance(wrappee, nn.BatchNorm2d):
        return _PnBatchNorm(wrappee)
    elif isinstance(wrappee, nn.Sequential):
        # Wrap each submodule
        for i, x in enumerate(wrappee):
            wrappee[i] = wrap(x)
    else:
        for submodule_name, submodule in wrappee.named_children():
            setattr(wrappee, submodule_name, wrap(submodule))
    return wrappee


class _TaskMaskParent(SurpriseNet, nn.Module):
    """
    _PackNetParent is used to apply PackNet methods to all of the child modules
    that implement PackNet

    :param PackNet: Inherit PackNet functionality
    :param nn.Module: Inherit ability to apply a function
    """

    def __init__(self) -> None:
        super().__init__()
        self._subset_count: Tensor
        self.register_buffer("_subset_count", torch.tensor(0))

    def _pn_apply(self, func: t.Callable[["SurpriseNet"], None]):
        @torch.no_grad()
        def __pn_apply(module):
            # Apply function to all child packnets but not other parents.
            # If we were to apply to other parents we would duplicate
            # applications to their children
            if isinstance(module, ParameterMask) and not isinstance(
                module, _TaskMaskParent
            ):
                func(module)

        self.apply(__pn_apply)

    def prune(self, to_prune_proportion: float) -> None:
        """
        Prunes the layer by removing the smallest weights and freezing them.
        Biases are frozen as a side-effect.
        """
        self._pn_apply(lambda x: x.prune(to_prune_proportion))

    def push_pruned(self) -> None:
        """
        Pushes the pruned weights to the next layer.
        """
        self._pn_apply(lambda x: x.push_pruned())
        self._subset_count += 1

    def mutable_activate_subsets(self, visible_subsets: t.List[int]):
        """
        Activates the subsets given subsets making them visible. The remaining
        capacity is mutable.
        """
        self._pn_apply(lambda x: x.mutable_activate_subsets(visible_subsets))

    def activate_subsets(self, subset_ids: t.List[int]):
        """
        Activates the given subsets in the layer making them visible. The
        remaining capacity is mutable.
        """
        self._pn_apply(lambda x: x.activate_subsets(subset_ids))

    def subset_count(self) -> int:
        return int(self._subset_count)


class SurpriseNetAutoEncoder(InferTask, AutoEncoder, _TaskMaskParent):
    """
    A wrapper for AutoEncoder adding the InferTask trait and PackNet
    functionality
    """

    def __init__(
        self,
        auto_encoder: AutoEncoder,
        task_inference_strategy: TaskInferenceStrategy,
        subset_activation_strategy: ActivationStrategy = NaiveSurpriseNetActivation(),
    ) -> None:
        super().__init__(
            auto_encoder._encoder, auto_encoder._decoder, auto_encoder._classifier
        )
        wrap(auto_encoder)
        self.subset_activation_strategy = subset_activation_strategy
        self.task_inference_strategy = task_inference_strategy

    def multi_forward(self, x: Tensor) -> ForwardOutput:
        if self.training:
            return super().multi_forward(x)
        else:
            """At eval time we need to try infer the task somehow?"""
            return self.task_inference_strategy.forward_with_task_inference(
                super().multi_forward, x
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.multi_forward(x).y_hat


class SurpriseNetVariationalAutoEncoder(
    InferTask, VariationalAutoEncoder, _TaskMaskParent, ConditionedSample
):
    """
    A wrapper for VariationalAutoEncoder adding the InferTask trait and PackNet
    functionality.
    """

    def __init__(
        self,
        auto_encoder: VariationalAutoEncoder,
        task_inference_strategy: TaskInferenceStrategy,
        subset_activation_strategy: ActivationStrategy = NaiveSurpriseNetActivation(),
    ) -> None:
        super().__init__(
            auto_encoder._encoder,
            auto_encoder.bottleneck,
            auto_encoder._decoder,
            auto_encoder._classifier,
        )
        wrap(auto_encoder)
        self.task_inference_strategy = task_inference_strategy
        self.subset_activation_strategy = subset_activation_strategy

    def multi_forward(self, x: Tensor) -> ForwardOutput:
        if self.training:
            return super().multi_forward(x)
        else:
            """At eval time we need to try infer the task somehow?"""
            return self.task_inference_strategy.forward_with_task_inference(
                super().multi_forward, x
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.multi_forward(x).y_hat

    def conditioned_sample(self, n: int = 1, given_task: int = 0) -> Tensor:
        self.activate_task_id(given_task)
        return self.sample(n)


class SurpriseNetDeepVAE(
    Classifier,
    InferTask,
    Encoder,
    Decoder,
    Samplable,
    MultiOutputNetwork,
    _TaskMaskParent,
):
    def __init__(
        self,
        wrapped: FashionMNISTDeepVAE,
        task_inference_strategy: TaskInferenceStrategy,
        subset_activation_strategy: ActivationStrategy = NaiveSurpriseNetActivation(),
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.task_inference_strategy = task_inference_strategy
        self.task_inference_strategy.model = self
        self.subset_activation_strategy = subset_activation_strategy
        wrap(wrapped)

    def forward(
        self,
        x: Tensor,
        n_posterior_samples: int = 1,
        use_mode: bool | t.List[bool] = False,
        decode_from_p: bool | t.List[bool] = False,
        **stage_kwargs: t.Any,
    ):
        return self.wrapped(
            x, n_posterior_samples, use_mode, decode_from_p, **stage_kwargs
        )

    def sample(self, n: int = 1) -> Tensor:
        return self.wrapped.sample(n)

    def decode(self, embedding: Tensor) -> Tensor:
        return self.wrapped.decode(embedding)

    def encode(self, x: Tensor) -> Tensor:
        return self.wrapped.encode(x)

    def classify(self, x: Tensor) -> Tensor:
        out = self.multi_forward(x)
        return out.y_hat

    def multi_forward(self, x: Tensor) -> ForwardOutput:
        if self.training:
            return self.wrapped.multi_forward(x)
        else:
            """At eval time we need to try infer the task somehow?"""
            return self.task_inference_strategy.forward_with_task_inference(
                self.wrapped.multi_forward, x
            )

    @property
    def n_latents(self) -> int:
        return self.wrapped.n_latents
