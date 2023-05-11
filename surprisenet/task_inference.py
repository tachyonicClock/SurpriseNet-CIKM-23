import typing as t
from dataclasses import fields

import torch
from experiment.experiment import BaseExperiment
from experiment.strategy import ForwardOutput
from network.hvae.oodd.losses import ELBO
from network.trait import PackNet
from torch import Tensor
from torch.nn import functional as F

if t.TYPE_CHECKING:
    from surprisenet.packnet import SurpriseNetDeepVAE, StageData


class TaskInferenceStrategy():
    def forward_with_task_inference(self,
                                    forward_func: t.Callable[[Tensor], ForwardOutput],
                                    x: Tensor) -> ForwardOutput:
        raise NotImplementedError()


class UseTaskOracle(TaskInferenceStrategy):
    """
    Cheat by using a task oracle
    """

    def __init__(self, experiment: BaseExperiment) -> None:
        super().__init__()
        self.experiment = experiment

    def forward_with_task_inference(
            self,
            forward_func: t.Callable[[Tensor], ForwardOutput],
            x: Tensor) -> ForwardOutput:
        model = self.experiment.strategy.model
        task_id = self.experiment.strategy.experience.current_experience
        assert isinstance(
            model, PackNet), "Task inference only works on PackNet"

        model.use_task_subset(min(task_id, model.subset_count() - 1))
        out: ForwardOutput = forward_func(x)
        out.pred_exp_id = (torch.ones((x.shape[0], 1)) * task_id).int()
        model.use_top_subset()
        return out


def _swap_fields(dest: ForwardOutput, src: ForwardOutput, swap_mask: Tensor):
    """
    For each `ForwardOutput` field move elements in the Tensors using the swap_mask.
    Move not performed if either the dest or the src have an empty field
    """
    def _swap_tensor_elements(dest_tensor: Tensor, src_tensor: Tensor, swap_mask: Tensor):
        assert dest_tensor.shape[0] == src_tensor.shape[0] == swap_mask.shape[0], \
            f"dest_tensor, src_tensor and swap_mask must have the same first dim, "\
            f"got {dest_tensor.shape}, {src_tensor.shape}, {swap_mask.shape}"

        dest_tensor[swap_mask] = src_tensor[swap_mask]

    for field in fields(ForwardOutput):
        dest_value: t.Any = getattr(dest, field.name)
        src_value: t.Any = getattr(src, field.name)
        if dest_value == None or src_value == None:
            # Skip
            continue
        elif field.name == "kl_divergences":
            # Swap each element of the field
            for i in range(len(dest_value)):
                _swap_tensor_elements(dest_value[i], src_value[i], swap_mask)
        elif isinstance(dest_value, Tensor):
            # Swap the whole tensor
            _swap_tensor_elements(dest_value, src_value, swap_mask)
        else:
            raise NotImplementedError(
                f"Unsupported type {type(dest_value)}")

class TaskReconstruction(TaskInferenceStrategy):
    """
    Use instance reconstruction to infer task
    """

    def __init__(self, experiment: BaseExperiment) -> None:
        super().__init__()
        self.experiment = experiment
        self.n_experiences = experiment.n_experiences

    def forward_with_task_inference(
            self,
            forward_func: t.Callable[[Tensor], ForwardOutput],
            x: Tensor) -> ForwardOutput:
        model = self.experiment.strategy.model
        assert isinstance(model, PackNet)

        # Initialize the best output using the first subset. Subsequent subsets
        # will be compared to this one
        model.use_task_subset(0)
        best_loss = torch.ones(x.shape[0]).to(x.device) * float('inf')
        best_loss, best_out = sample(forward_func, x)
        best_out.pred_exp_id = torch.zeros(x.shape[0]).int()

        # Iterate over all subsets and compare them to the best subset
        for i in range(1, model.subset_count()):
            model.use_task_subset(i)
            new_loss, new_out = sample(forward_func, x)

            # Update best_out if the current subset is better
            swap_mask = new_loss < best_loss
            best_loss[swap_mask] = new_loss[swap_mask]
            best_out.pred_exp_id[swap_mask] = i
            _swap_fields(best_out, new_out, swap_mask)

        model.use_top_subset()
        return best_out


@torch.no_grad()
def sample(forward_func: t.Callable[[Tensor], ForwardOutput],
           x: Tensor) -> t.Tuple[Tensor, ForwardOutput]:
    out: ForwardOutput = forward_func(x)
    # NOTE: The reduction is done manually to allow for per-instance losses
    loss = F.mse_loss(out.x_hat, x, reduction='none').flatten(1).mean(1)
    assert loss.shape[0] == x.shape[0]
    return loss, out


class HierarchicalVAEOOD(TaskInferenceStrategy):
    """
    Uses the Hierarchical VAE outlier detection method to infer the task
    Havtorn, J. D., Frellsen, J., Hauberg, S., & Maaløe, L. (n.d.). Hierarchical
    VAEs Know What They Don't Know.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        # TODO: Relying on model being set later by the client is spaghetti
        self.elbo_loss_func = ELBO()

    @staticmethod
    def _get_decode_from_p(n_latents, k=0, semantic_k=True):
        """
        k semantic out
        0 True     [False, False, False]
        1 True     [True, False, False]
        2 True     [True, True, False]
        0 False    [True, True, True]
        1 False    [False, True, True]
        2 False    [False, False, True]
        """
        if semantic_k:
            return [True] * k + [False] * (n_latents - k)

        return [False] * (k + 1) + [True] * (n_latents - k - 1)
    
    @staticmethod
    def _kl_divergences_from_stage(stage_datas: t.List['StageData']):
        return [
            stage_data.loss.kl_elementwise
            for stage_data in stage_datas
            if stage_data.loss.kl_elementwise is not None
        ]
    
    @torch.no_grad()
    def _novelty_score(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        # Regular ELBO
        likelihood_data, stage_data = self.model(x)
        _, elbo, _, _ = self.elbo_loss_func(
            likelihood_data.likelihood,
            self._kl_divergences_from_stage(stage_data),
            samples=1,
            free_nats=0,
            beta=1,
            sample_reduction=None,
            batch_reduction=None,
        )

        # L>k bound
        latent_hierarchy_depth = self.model.wrapped.deep_vae.n_latents
        decode_from_p = self._get_decode_from_p(latent_hierarchy_depth, k=1)
        likelihood_data, stage_data = self.model(x, decode_from_p=decode_from_p, use_mode=decode_from_p)
        _, elbo_k, _, _ = self.elbo_loss_func(
            likelihood_data.likelihood,
            self._kl_divergences_from_stage(stage_data),
            samples=1,
            free_nats=0,
            beta=1,
            sample_reduction=None,
            batch_reduction=None,
        )

        return elbo - elbo_k
    
    def sample_score(self, model, forward_func, x):
        out: ForwardOutput = forward_func(x)
        score = self._novelty_score(x)
        return score, out


    def forward_with_task_inference(
            self,
            forward_func: t.Callable[[Tensor], ForwardOutput],
            x: Tensor) -> ForwardOutput:

        # Initialize the best output using the first subset. Subsequent subsets
        # will be compared to this one
        self.model.use_task_subset(0)
        best_score = torch.ones(x.shape[0]).to(x.device) * float('inf')
        best_score, best_out = self.sample_score(self.model, forward_func, x)
        best_out.pred_exp_id = torch.zeros(x.shape[0]).int()

        # Iterate over all subsets and compare them to the best subset
        for i in range(1, self.model.subset_count()):
            self.model.use_task_subset(i)
            new_score, new_out = self.sample_score(self.model, forward_func, x)

            # Update best_out if the current subset is better
            swap_mask = new_score < best_score
            best_score[swap_mask] = new_score[swap_mask]
            best_out.pred_exp_id[swap_mask] = i
            _swap_fields(best_out, new_out, swap_mask)

        self.model.use_top_subset()
        return best_out
