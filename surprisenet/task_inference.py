import typing as t
from dataclasses import fields

import torch
from experiment.experiment import BaseExperiment
from experiment.strategy import ForwardOutput
from network.trait import PackNet
from torch import Tensor
from torch.nn import functional as F


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
