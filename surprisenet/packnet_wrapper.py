import math
import typing as t
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from network.trait import PackNet


class ModuleDecorator(nn.Module):
    wrappee: nn.Module

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee.forward(input)

    def __init__(self, wrappee: nn.Module):
        super().__init__()
        self.add_module("wrappee", wrappee)


class StateError(Exception):
    pass


class PackNetDecorator(PackNet, ModuleDecorator):
    """
    PackNetDecorator implement PackNet functionality for a supplied weight buffer.
    You can think about a PackNet as a Stack of networks overlaid on top of each other
    where each layer (in the sense of the stack) is only connected to those
    below additionally only the top of the stack can be trained.

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
        """Use a non-top layer of the network. In this state training cannot be
          conducted"""

    def _state_guard(self, previous: t.Sequence[State], next: State):
        """Ensure that the state is in the correct state and transition to the
        next correct state. If the state is not in the correct state then raise
        a StateError. This ensures that the correct procedure is followed.
        """

        if self.state not in previous:
            raise StateError(
                f"Function only valid for {previous} instead PackNet was "
                + "in the {self.state} state"
            )
        self.state = next

    @property
    def top_mask(self) -> Tensor:
        """Return a mask of weights at the top of the `stack`"""
        return self.task_index.eq(self._z_top)

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.task_index.eq(self._Z_PRUNED)

    @property
    def weight(self) -> Tensor:
        return NotImplemented

    @property
    def bias(self) -> Tensor:
        return NotImplemented

    def _remove_gradient_hook(self, grad: Tensor) -> Tensor:
        """
        Only the top layer should be trained. Todo so all other gradients
        are zeroed. Caution should be taken when optimizers with momentum are
        used, since they can cause parameters to be modified even when no
        gradient exists
        """
        grad = grad.clone()
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
        return rank[un_prunable.count_nonzero() :]

    def _prune_weights(self, indices: Tensor):
        self.task_index.flatten()[indices] = self._Z_PRUNED.item()

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
        self._state_guard([self.State.MUTABLE_TOP], self.State.PRUNED_TOP)
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * to_prune_proportion)
        self._prune_weights(ranked[:prune_count])

    def available_weights(self) -> Tensor:
        if self.state == self.state.MUTABLE_TOP:
            return self.weight
        weight = self.weight.clone()
        # Mask of the weights that are above the supplied z_index. Used to zero
        # them
        mask = self.task_index.greater(self._z_active)
        with torch.no_grad():
            weight[mask] = 0.0
        return self.visible_mask * self.weight

    def use_task_subset(self, subset_id):
        """Setter to set the sub-set of the network to be used on forward pass"""
        assert (
            subset_id >= 0 and subset_id <= self._z_top
        ), f"subset_id {subset_id} must be between 0 and {self._z_top}"

        next_state = (
            self.State.MUTABLE_TOP if self._z_top == subset_id else self.State.IMMUTABLE
        )
        self._state_guard([self.State.MUTABLE_TOP, self.State.IMMUTABLE], next_state)

        subset_id = min(max(subset_id, 0), self._z_top)
        self._z_active.fill_(subset_id)

        # self.visible_mask = self.task_index.less(self._z_active)

    def use_top_subset(self):
        """Forward should use the top subset"""
        self.use_task_subset(self._z_top)

    def initialize_top(self):
        """Re-initialize the top of the network"""
        # He Weight Initialization
        stddev = math.sqrt(2 / self.top_mask.count_nonzero())
        dist = torch.distributions.Normal(0, stddev)
        with torch.no_grad():
            self.weight[self.top_mask] = dist.sample(
                (self.top_mask.count_nonzero(),)
            ).to(self.device)

    def push_pruned(self):
        self._state_guard([self.State.PRUNED_TOP], self.State.MUTABLE_TOP)
        # The top is now one higher up
        self._z_top += 1
        # Move pruned weights to the top
        self.task_index[self.pruned_mask] = self._z_top.item()
        # Change the active z_index
        self.use_top_subset()
        self.initialize_top()

        if self.bias is not None:
            self.bias.requires_grad = False

    def subset_count(self) -> int:
        return int(self._z_top.item())

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__(wrappee)

        self._z_top: torch.Tensor
        """Index top of the 'stack'. Should only increase"""
        self._z_active: torch.Tensor
        """
        Index defining which subset is used for a forward pass
        """
        self._Z_PRUNED: torch.Tensor
        """Index tracking if a weight has been pruned"""
        self.task_index: torch.Tensor
        """
        Z mask is a depth index of weights in an imaginary "stack" that makes up the
        PackNet. The masks values corresponds to each task.
        """

        self.visible_mask: torch.Tensor
        self.gradient_mask: torch.Tensor

        # Register buffers. Buffers are tensors that are not parameters, but
        # should be saved with the model.
        self.register_buffer("_z_top", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_z_active", torch.tensor(0, dtype=torch.int))
        self.register_buffer("_Z_PRUNED", torch.tensor(255, dtype=torch.int))
        self.register_buffer(
            "task_index", torch.ones(self.weight.shape).byte() * self._z_top
        )
        self.register_buffer(
            "_state", torch.tensor(self.State.MUTABLE_TOP.value, dtype=torch.int)
        )
        """The state of the PackNet used to avoid getting into invalid states"""

        self.visible_mask = torch.ones_like(self.weight, dtype=torch.bool)

        assert self.weight != NotImplemented
        assert self.bias != NotImplemented
        self.weight.register_hook(self._remove_gradient_hook)

    @property
    def state(self) -> State:
        return self.State(self._state.item())

    @state.setter
    def state(self, state: State):
        self._state.fill_(state.value)
