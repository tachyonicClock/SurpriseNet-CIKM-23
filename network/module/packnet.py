from abc import abstractmethod
from curses import wrapper
import math
from sre_parse import State
from turtle import forward
import typing as t

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from enum import Enum
from experiment.strategy import ForwardOutput
from experiment.task_inference import TaskInferenceStrategy

from network.trait import AutoEncoder, InferTask, PackNet, VariationalAutoEncoder


class ModuleDecorator(nn.Module):
    wrappee: nn.Module

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee.forward(input)

    def __init__(self, wrappee: nn.Module):
        super().__init__()
        self.add_module('wrappee', wrappee)


class PackNetDecorator(PackNet, ModuleDecorator):
    """
    PackNetDecorator implement PackNet functionality for a supplied weight buffer. 
    You can think about a PackNet as a Stack of networks overlayed on top of each other
    where each layer is only connected to those below additionally only the
    top of the stack can be trained.

    In order to grow the stack the top must be pruned `PackNetStack.prune` to
    free up parameters. The pruned parameters then need to pushed to the top of
    the stack with `PackNetStack.push_pruned()`


    Bib:
    Mallya, A., & Lazebnik, S. (2018). PackNet: Adding Multiple Tasks to a 
    Single Network by Iterative Pruning. 2018 IEEE/CVF Conference on Computer 
    Vision and Pattern Recognition, 7765-7773. 
    https://doi.org/10.1109/CVPR.2018.00810

    Han, S., Pool, J., Narang, S., Mao, H., Gong, E., Tang, S., Elsen, E., 
    Vajda, P., Paluri, M., Tran, J., Catanzaro, B., & Dally, W. J. (2017). 
    DSD: Dense-Sparse-Dense Training for Deep Neural Networks. 
    ArXiv:1607.04381 [Cs]. http://arxiv.org/abs/1607.04381
    """

    class State(Enum):
        """
        PackNet requires a procedure to be followed and we model this with the
        following states
        """
        MUTABLE_TOP = 0
        """Normally train the top of the network"""
        PRUNED_TOP = 1
        """Post prune training"""
        IMMUTABLE = 2
        """Use a non-top layer of the network. In this state training cannot be conducted"""

    class StateError(Exception):
        pass

    def next_state(self, previous: t.Sequence[State], next: State):
        if self.state not in previous:
            raise self.StateError(
                f"Function only valid for {previous} instead PackNet was in the {self.state} state")
        self.state = next

    # TODO Rename to supermask
    z_mask: Tensor
    """
    Z mask is a depth index of weights in an imaginary "stack" that makes up the
    PackNet. The masks values corresponds to each task.
    """

    _Z_PRUNED = 255
    """Index tracking if a weight has been pruned"""
    _z_top: int = 0
    """Index top of the 'stack'. Should only increase"""
    _z_active: int = 0
    """
    Index defining which subset is used for a forward pass
    """

    @property
    def top_mask(self) -> Tensor:
        """Return a mask of weights at the top of the `stack`"""
        return self.z_mask.eq(self._z_top)

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.z_mask.eq(self._Z_PRUNED)

    @property
    def weight(self) -> Tensor:
        return NotImplemented

    @property
    def bias(self) -> Tensor:
        return NotImplemented

    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__(wrappee)
        self.register_buffer("z_mask", torch.ones(
            self.weight.shape).byte() * self._z_top)
        self.state = self.State.MUTABLE_TOP

        assert self.weight != NotImplemented, "Concrete decorator must implement self.weight"
        assert self.bias != NotImplemented, "Concrete decorator must implement self.bias"
        self.weight.register_hook(self._remove_gradient_hook)

        # if self.bias != None:
        #     self.bias.requires_grad = False

    def _remove_gradient_hook(self, grad: Tensor) -> Tensor:
        """
        Only the top layer should be trained. Todo so all other gradients
        are zeroed. Caution should be taken when optimizers with momentum are 
        used, since they can cause parameters to be modified even when no 
        gradient exists
        """
        grad[~self.top_mask] = 0.0
        return grad

    def _rank_prunable(self) -> Tensor:
        """
        Returns a 1D tensor of the weights ranked based on their absolute value.
        Sorted to be in ascending order.
        """
        # "We use the simple heuristic to quantify the importance of the
        # weights using their absolute value." (Han et al., 2017)
        importance = self.weight.abs()

        # All weights that are not on top of the stack are un_prunable
        un_prunable = ~self.top_mask
        # Mark un-prunable weights using -1.0 so they can be cutout after sort
        importance[un_prunable] = -1.0

        # Rank the importance
        rank = torch.argsort(importance.flatten())
        # Cut out un-prunable weights
        return rank[un_prunable.count_nonzero():]

    def _prune_weights(self, indices: Tensor):
        self.z_mask.flatten()[indices] = self._Z_PRUNED

        # "Weight initialization plays a big role in deep learning (Mishkin &
        # Matas (2015)). Conventional training has only one chance of
        # initialization. DSD gives the optimization a second (or more) chance
        # during the training process to re-initialize from more robust sparse
        # training solution. We re-dense the network from the sparse solution
        # which can be seen as a zero initialization for pruned weights. Other
        # initialization methods are also worth trying." (Han et al., 2017)
        with torch.no_grad():
            self.weight.flatten()[indices] = 0.0

    def prune(self, to_prune_proportion: float):
        self.next_state([self.State.MUTABLE_TOP], self.State.PRUNED_TOP)
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * to_prune_proportion)
        self._prune_weights(ranked[:prune_count])

    def available_weights(self) -> Tensor:
        if self.state == self.state.MUTABLE_TOP:
            return self.weight
        weight = self.weight.clone()
        # Mask of the weights that are above the supplied z_index. Used to zero
        # them
        mask = self.z_mask.greater(self._z_active)
        with torch.no_grad():
            weight[mask] = 0.0
        return weight

    def use_task_subset(self, task_id):
        """Setter to set the sub-set of the network to be used on forward pass"""
        next_state = self.State.MUTABLE_TOP if self._z_top == task_id else self.State.IMMUTABLE
        self.next_state(
            [self.State.MUTABLE_TOP, self.State.IMMUTABLE], next_state)

        task_id = min(max(task_id, 0), self._z_top)
        self._z_active = task_id

    def use_top_subset(self):
        """Forward should use the top subset"""
        self.use_task_subset(self._z_top)

    def initialize_top(self):
        """Re-initialize the top of the network"""
        # He Weight Initialization
        stddev = math.sqrt(2/self.top_mask.count_nonzero())
        dist = torch.distributions.Normal(0, stddev)
        with torch.no_grad():
            self.weight[self.top_mask] = dist.sample(
                (self.top_mask.count_nonzero(),)).to(self.weight.device)

    def push_pruned(self):
        self.next_state([self.State.PRUNED_TOP], self.State.MUTABLE_TOP)
        # The top is now one higher up
        self._z_top += 1
        # Move pruned weights to the top
        self.z_mask[self.pruned_mask] = self._z_top
        # Change the active z_index
        self.use_top_subset()

        self.initialize_top()

        if self.bias != None:
            self.bias.requires_grad = False


class _PnBatchNorm(PackNet, ModuleDecorator):
    """BatchNorm is insanely annoying 
    """

    frozen: bool = False

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
        pass

    def use_task_subset(self, task_id):
        pass

    def use_top_subset(self):
        pass


class _PnLinear(PackNetDecorator):
    wrappee: nn.Linear

    def __init__(self, wrappee: nn.Linear) -> None:
        super().__init__(wrappee)
        self.weight_count = self.wrappee.in_features * self.wrappee.out_features

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


class _PnConv2d(PackNetDecorator):

    wrappee: nn.Conv2d

    def __init__(self, wrappee: nn.Conv2d) -> None:
        super().__init__(wrappee)

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    def initialize_top(self):
        # He Weight Initialization
        stddev = math.sqrt(2/self.top_mask.count_nonzero())
        dist = torch.distributions.Normal(0, stddev)
        with torch.no_grad():
            self.weight[self.top_mask] = dist.sample(
                (self.top_mask.count_nonzero(),)).to(self.weight.device)

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee._conv_forward(input, self.available_weights(), self.bias)


class _PnConvTransposed2d(PackNetDecorator):
    wrappee: nn.ConvTranspose2d

    def __init__(self, wrappee: nn.ConvTranspose2d) -> None:
        super().__init__(wrappee)

    @property
    def weight(self) -> Tensor:
        return self.wrappee.weight

    @property
    def bias(self) -> Tensor:
        return self.wrappee.bias

    def forward(self, input: Tensor, output_size: t.Optional[t.List[int]] = None) -> Tensor:
        w = self.wrappee
        if w.padding_mode != 'zeros':
            raise ValueError(
                'Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(w.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = w._output_padding(
            input, output_size, w.stride, w.padding, w.kernel_size, w.dilation)  # type: ignore[arg-type]

        return F.conv_transpose2d(
            input, self.available_weights(), w.bias, w.stride, w.padding,
            output_padding, w.groups, w.dilation)


def wrap(wrappee: nn.Module):
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


def deffer_wrap(wrappee: t.Type[nn.Module]):
    def _deffered(*args, **kwargs):
        return wrap(wrappee(*args, **kwargs))
    return _deffered


class _PackNetParent(PackNet, nn.Module):
    """
    _PackNetParent is used to apply PackNet methods to all of the child modules
    that implement PackNet

    :param PackNet: Inherit PackNet functionality
    :param nn.Module: Inherit ability to apply a function
    """

    def _pn_apply(self, func: t.Callable[['PackNet'], None]):
        @torch.no_grad()
        def __pn_apply(module):
            # Apply function to all child packnets but not other parents.
            # If we were to apply to other parents we would duplicate
            # applications to their children
            if isinstance(module, PackNet) and not isinstance(module, _PackNetParent):
                func(module)

        self.apply(__pn_apply)

    def prune(self, to_prune_proportion: float) -> None:
        self._pn_apply(lambda x: x.prune(to_prune_proportion))

    def push_pruned(self) -> None:
        self._pn_apply(lambda x: x.push_pruned())

    def use_task_subset(self, task_id):
        self._pn_apply(lambda x: x.use_task_subset(task_id))

    def use_top_subset(self):
        self._pn_apply(lambda x: x.use_top_subset())


class PackNetAutoEncoder(InferTask, AutoEncoder, _PackNetParent):
    """
    A wrapper for AutoEncoder adding the InferTask trait and PackNet
    functionality
    """
    def __init__(self,
                 auto_encoder: AutoEncoder,
                 task_inference_strategy: TaskInferenceStrategy) -> None:
        super().__init__(auto_encoder.encoder, auto_encoder.decoder, auto_encoder.classifier)
        wrap(auto_encoder)
        self.task_inference_strategy = task_inference_strategy

    def forward(self, x: Tensor) -> ForwardOutput:

        if self.training:
            return super().forward(x)
        else:
            """At eval time we need to try infer the task somehow?"""
            return self.task_inference_strategy \
                       .forward_with_task_inference(super().forward, x)


class PackNetVariationalAutoEncoder(InferTask, VariationalAutoEncoder, _PackNetParent):
    """
    A wrapper for VariationalAutoEncoder adding the InferTask trait and PackNet
    functionality.
    """

    def __init__(self,
                 auto_encoder: VariationalAutoEncoder,
                 task_inference_strategy: TaskInferenceStrategy) -> None:
        super().__init__(auto_encoder.encoder, auto_encoder.bottleneck, auto_encoder.decoder, auto_encoder.classifier)
        wrap(auto_encoder)
        self.task_inference_strategy = task_inference_strategy

    def forward(self, x: Tensor) -> ForwardOutput:

        if self.training:
            return super().forward(x)
        else:
            """At eval time we need to try infer the task somehow?"""
            return self.task_inference_strategy \
                       .forward_with_task_inference(super().forward, x)
