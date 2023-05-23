import typing as t
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from network.trait import ParameterMask


class ModuleDecorator(nn.Module):
    wrappee: nn.Module

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee.forward(input)

    def __init__(self, wrappee: nn.Module):
        super().__init__()
        self.add_module("wrappee", wrappee)


class StateError(Exception):
    pass


class WeightMask(ParameterMask, ModuleDecorator):
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
                + f"in the {self.state} state"
            )
        self.state = next

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.task_index.eq(self.PRUNED_CODE)

    @property
    def weight(self) -> Tensor:
        return NotImplemented

    @property
    def bias(self) -> Tensor:
        return NotImplemented

    @property
    def state(self) -> State:
        return self.State(self._state.item())

    @state.setter
    def state(self, state: State):
        self._state.fill_(state.value)

    def _remove_gradient_hook(self, grad: Tensor) -> Tensor:
        """
        Only the top layer should be trained. Todo so all other gradients
        are zeroed. Caution should be taken when optimizers with momentum are
        used, since they can cause parameters to be modified even when no
        gradient exists
        """
        return grad * self.mutability_mask

    def _rank_prunable(self) -> Tensor:
        """
        Returns a 1D tensor of the weights ranked based on their absolute value.
        Sorted to be in ascending order.
        """
        # "We use the simple heuristic to quantify the importance of the
        # weights using their absolute value." (Han et al., 2017)
        importance = self.weight.abs()
        un_prunable = ~self.mutability_mask
        # Mark un-prunable weights using -1.0 so they can be cutout after sort
        importance[un_prunable] = -1.0
        # Rank the importance
        rank = torch.argsort(importance.flatten())
        # Cut out un-prunable weights
        return rank[un_prunable.count_nonzero() :]

    def _prune_weights(self, indices: Tensor):
        self.task_index.flatten()[indices] = self.PRUNED_CODE.item()
        self.visiblity_mask.flatten()[indices] = False

    def prune(self, to_prune_proportion: float):
        self._state_guard([self.State.MUTABLE_TOP], self.State.PRUNED_TOP)
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * to_prune_proportion)
        self._prune_weights(ranked[:prune_count])
        self.mutability_mask = self.task_index.eq(self._subset_count)

    def available_weights(self) -> Tensor:
        return self.visiblity_mask * self.weight

    def _is_subset_id_valid(self, subset_id: t.List[int]):
        assert (
            0 <= subset_id <= self._subset_count
        ), f"Given Subset ID {subset_id} must be between 0 and {self._subset_count}"

    def mutable_activate_subsets(self, visible_subsets: t.List[int]):
        self.activate_subsets(visible_subsets)
        self._state_guard([self.State.IMMUTABLE], self.State.MUTABLE_TOP)

        self.mutability_mask = self.task_index.eq(self._subset_count)
        self.visiblity_mask = self.visiblity_mask | self.mutability_mask

    def activate_subsets(self, visible_subsets: t.List[int]):
        self._state_guard(
            [self.State.IMMUTABLE, self.State.MUTABLE_TOP], self.State.IMMUTABLE
        )
        self.visiblity_mask.zero_()
        for subset_id in visible_subsets:
            self._is_subset_id_valid(subset_id)
            self.visiblity_mask = self.visiblity_mask | self.task_index.eq(subset_id)

    def push_pruned(self):
        self._state_guard([self.State.PRUNED_TOP], self.State.IMMUTABLE)
        # The top is now one higher up
        self._subset_count += 1
        # Move pruned weights to the top
        self.task_index[self.pruned_mask] = self._subset_count.item()
        # Change the active z_index
        self.mutability_mask.zero_()

        if self.bias is not None:
            self.bias.requires_grad = False

    def subset_count(self) -> int:
        return int(self._subset_count.item())

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def __init__(self, wrappee: nn.Module) -> None:
        super().__init__(wrappee)
        self._subset_count: torch.Tensor
        """Index top of the 'stack'. Should only increase"""
        self.PRUNED_CODE: torch.Tensor
        """Scalar denoting the code for pruned weights"""
        self.task_index: torch.Tensor
        """Index of the task each weight belongs to"""
        self.visiblity_mask: torch.Tensor
        """Mask of weights that are presently visible"""
        self.mutability_mask: torch.Tensor
        """Mask of weights that are mutable"""

        # Register buffers. Buffers are tensors that are not parameters, but
        # should be saved with the model.
        self.register_buffer("_subset_count", torch.tensor(0, dtype=torch.int))
        self.register_buffer("PRUNED_CODE", torch.tensor(255, dtype=torch.int))
        self.register_buffer(
            "_state", torch.tensor(self.State.MUTABLE_TOP.value, dtype=torch.int)
        )
        self.register_buffer(
            "task_index", torch.ones(self.weight.shape).byte() * self._subset_count
        )
        self.register_buffer(
            "visiblity_mask", torch.ones_like(self.weight, dtype=torch.bool)
        )
        self.register_buffer(
            "mutability_mask", torch.ones_like(self.weight, dtype=torch.bool)
        )

        assert self.weight != NotImplemented
        assert self.bias != NotImplemented
        self.weight.register_hook(self._remove_gradient_hook)
