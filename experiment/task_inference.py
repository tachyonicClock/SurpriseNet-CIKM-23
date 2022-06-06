from abc import ABC, abstractclassmethod
import typing as t
from dataclasses import fields

from torch import Tensor, nn
import torch
from experiment.experiment import BaseExperiment
from experiment.loss import ReconstructionLoss

from experiment.strategy import ForwardOutput, Strategy
from metrics.metrics import TaskInferenceMetrics
from network.trait import AutoEncoder, PackNet
from torch.nn import functional as F


class TaskInferenceStrategy(ABC):

    @abstractclassmethod
    def forward_with_task_inference(self, 
            forward_func: t.Callable[[Tensor], ForwardOutput] ,
            x: Tensor) -> ForwardOutput:
        pass


class UseTaskOracle(TaskInferenceStrategy):
    """
    Cheat by using a task oracle
    """

    def __init__(self, experiment: BaseExperiment) -> None:
        super().__init__()
        self.experiment = experiment

    def forward_with_task_inference(self, 
            forward_func: t.Callable[[Tensor], ForwardOutput] ,
            x: Tensor) -> ForwardOutput:
        model = self.experiment.strategy.model
        task_id = self.experiment.strategy.experience.current_experience
        assert isinstance(model, PackNet), "Task inference only works on PackNet"

        model.use_task_subset(task_id)
        out : ForwardOutput = forward_func(x)
        out.pred_exp_id = (torch.ones((x.shape[0], 1)) * task_id).int()
        model.use_top_subset()
        return out


def _move_some(dest: ForwardOutput, src: ForwardOutput, swap_mask: Tensor):
    """
    For each `ForwardOutput` field move elements in the Tensors using the swap_mask.
    Move not performed if either the dest or the src have an empty field
    """
    for field in fields(ForwardOutput):
        dest_tensor: Tensor = getattr(dest, field.name)
        src_tensor: Tensor = getattr(src, field.name)
        if dest_tensor == None or src_tensor == None:
            continue
        dest_tensor[swap_mask] = src_tensor[swap_mask]

class TaskReconstruction(TaskInferenceStrategy):
    """
    Use instance reconstruction to infer task
    """

    def __init__(self, experiment: BaseExperiment) -> None:
        super().__init__()
        self.experiment = experiment
        self.n_experiences = experiment.n_experiences

        experiment.add_plugin(
            TaskInferenceMetrics(experiment.logdir)
        )

    def forward_with_task_inference(self, 
            forward_func: t.Callable[[Tensor], ForwardOutput],
            x: Tensor) -> ForwardOutput:
        # Loss for each instance for each layer
        loss_by_layer = torch.zeros((self.n_experiences, x.shape[0]))

        model = self.experiment.strategy.model
        sample_size = 1
        assert isinstance(model, PackNet), "Task inference only works on PackNet"
        model.use_task_subset(0)

        best_loss, best_out = sample(forward_func, x, sample_size)
        loss_by_layer[0, :] = best_loss
        best_out.pred_exp_id = torch.zeros(x.shape[0]).int().to(best_out.x_hat.device)


        for i in range(1, self.n_experiences):
            # Use a specific subset
            model.use_task_subset(i)

            new_loss, new_out = sample(forward_func, x, sample_size)
            loss_by_layer[i, :] = new_loss

            # Update best_out if the current subset is better
            swap_mask = new_loss < best_loss
            # Manually set the predicted experience
            best_out.pred_exp_id[swap_mask] = i
            # Move all other fields automatically
            _move_some(best_out, new_out, swap_mask)
            # Ensure that best_loss is always the best possible loss
            best_loss[swap_mask] = new_loss[swap_mask]

        model.use_top_subset()
        best_out.loss_by_layer = loss_by_layer
        return best_out

def bce_task_inference_loss(input: Tensor, target: Tensor) -> Tensor:
    bce: Tensor = F.binary_cross_entropy(input, target, reduction="none")
    # Mean each 
    return torch.mean(bce, dim=(1,2,3))


# TODO this is not the final sampling logic and needs work. We should
# probably use the mean
@torch.no_grad()
def sample(
    forward_func: t.Callable[[Tensor], ForwardOutput],
    x: Tensor,
    sample_size: int) -> t.Tuple[Tensor, ForwardOutput]:

    best_out: ForwardOutput = forward_func(x)
    best_loss: Tensor = bce_task_inference_loss(best_out.x_hat, x)
    loss_total: Tensor = best_loss.detach().clone()

    for i in range(1, sample_size):
        new_out: ForwardOutput = forward_func(x)
        new_loss = bce_task_inference_loss(best_out.x_hat, x)
        loss_total += new_loss

        # Update best_out if the current subset is better
        swap_mask = new_loss < best_loss
        # Move all other fields automatically
        _move_some(best_out, new_out, swap_mask)
        # Ensure that best_loss is always the best possible loss
        best_loss[swap_mask] = new_loss[swap_mask]

    return loss_total/sample_size, best_out