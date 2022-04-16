from dataclasses import fields
import typing
import torch
from torch import Tensor
from experiment.strategy import ForwardOutput, Network
from functional import MRAE
from network.trait import AutoEncoder, PackNet


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

@torch.no_grad()
def infer_task(
    ae: typing.Union[AutoEncoder, PackNet, Network],
    batch: Tensor,
    task_count: int,
    sample_size: int = 1) -> ForwardOutput:
    """_summary_

    :param ae: _description_
    :param batch: _description_
    :param task_count: _description_
    :return: _description_
    """

    ae.use_task_subset(0)
    best_loss, best_out = sample(ae, batch, sample_size)
    best_out.pred_exp_id = torch.zeros(batch.shape[0]).int().to(best_out.x_hat.device)

    for i in range(1, task_count):
        # Use a specific subset
        ae.use_task_subset(i)

        # new_out: ForwardOutput = ae.forward(batch)
        # new_loss: Tensor = MRAE(new_out.x_hat, batch, reduce_batch=False)
        new_loss, new_out = sample(ae, batch, sample_size)

        # Update best_out if the current subset is better
        swap_mask = new_loss < best_loss
        # Manually set the predicted experience
        best_out.pred_exp_id[swap_mask] = i
        # Move all other fields automatically
        _move_some(best_out, new_out, swap_mask)
        # Ensure that best_loss is always the best possible loss
        best_loss[swap_mask] = new_loss[swap_mask]

    ae.use_top_subset()
    return best_out

@torch.no_grad()
def sample(
    ae: typing.Union[AutoEncoder, PackNet, Network],
    batch: Tensor,
    sample_size: int) -> typing.Tuple[Tensor, ForwardOutput]:

    best_out: ForwardOutput = ae.forward(batch)
    best_loss: Tensor = MRAE(best_out.x_hat, batch, reduce_batch=False)
    loss_total: Tensor = best_loss.detach().clone()

    for i in range(1, sample_size):
        new_out: ForwardOutput = ae.forward(batch)
        new_loss = MRAE(new_out.x_hat, batch, reduce_batch=False)
        loss_total += new_loss

        # Update best_out if the current subset is better
        swap_mask = new_loss < best_loss
        # Move all other fields automatically
        _move_some(best_out, new_out, swap_mask)
        # Ensure that best_loss is always the best possible loss
        best_loss[swap_mask] = new_loss[swap_mask]

    return loss_total/sample_size, best_out




