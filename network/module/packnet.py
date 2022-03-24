from abc import abstractmethod
import math
from sre_parse import State
import typing

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from enum import Enum

from network.trait import PackNet

class ModuleDecorator(nn.Module):
    wrappee: nn.Module

    def forward(self, input: Tensor) -> Tensor:
        return self.wrappee.forward(input)

    def __init__(self, wrappee: nn.Module):
        super().__init__()
        self.add_module('wrappee', wrappee)


class PackNetDecorator(PackNet, ModuleDecorator):
    """
    PackNetStack implement PackNet functionality for a supplied weight buffer. 
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

    def next_state(self, previous: typing.Sequence[State], next: State):
        if self.state not in previous:
            raise self.StateError(f"Function only valid for {previous} instead PackNet was in the {self.state} state")
        self.state = next

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
        self.register_buffer("z_mask", torch.ones(self.weight.shape).byte() * self._z_top)
        self.state = self.State.MUTABLE_TOP

        assert self.weight != NotImplemented, "Concrete decorator must implement self.weight"
        assert self.bias != NotImplemented, "Concrete decorator must implement self.bias"
        self.weight.register_hook(self._remove_gradient_hook)

        self.bias.requires_grad = False


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
        with torch.no_grad(): self.weight.flatten()[indices] = 0.0

    def prune(self, to_prune_proportion: float):
        self.next_state([self.State.MUTABLE_TOP], self.State.PRUNED_TOP)
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * to_prune_proportion)
        self._prune_weights(ranked[:prune_count])

    def available_weights(self) -> Tensor:
        weight = self.weight.clone()
        # Mask of the weights that are above the supplied z_index. Used to zero
        # them 
        mask = self.z_mask.greater(self._z_active)
        with torch.no_grad(): weight[mask] = 0.0
        return weight

    def use_task_subset(self, task_id):
        """Setter to set the sub-set of the network to be used on forward pass"""
        next_state = self.State.MUTABLE_TOP if self._z_top == task_id else self.State.IMMUTABLE
        self.next_state([self.State.MUTABLE_TOP, self.State.IMMUTABLE], next_state)

        task_id = min(max(task_id, 0), self._z_top)
        self._z_active = task_id

    def use_top_subset(self):
        """Forward should use the top subset"""
        self.use_task_subset(self._z_top)

    def push_pruned(self):
        self.next_state([self.State.PRUNED_TOP], self.State.MUTABLE_TOP)
        # The top is now one higher up
        self._z_top += 1
        # Move pruned weights to the top
        self.z_mask[self.pruned_mask] = self._z_top
        # Change the active z_index
        self.use_top_subset()

        if self.bias != None:
            self.bias.requires_grad = False

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

    def forward(self, input: Tensor, output_size: typing.Optional[typing.List[int]] = None) -> Tensor:
        w = self.wrappee
        if w.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

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
    elif isinstance(wrappee, nn.Sequential):
        # Wrap each submodule
        for i, x in enumerate(wrappee):
            wrappee[i] = wrap(x)
    return wrappee

def deffer_wrap(wrappee: typing.Type[nn.Module]):
    def _deffered(*args, **kwargs):
        return wrap(wrappee(*args, **kwargs))
    return _deffered


