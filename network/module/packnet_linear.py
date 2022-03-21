import math
import typing

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from network.trait import PackNetModule

class PackNetLinear(PackNetModule):
    """
    Implementation of linear module to implement PackNet functionality. You
    can think about a PackNet as a Stack of networks overlayed on top of each other
    where each layer is only connected to those below additionally only the
    top of the stack can be trained.

    In order to grow the stack the top must be pruned `PackNetLinear.prune` to
    free up parameters. The pruned parameters then need to pushed to the top of
    the stack with `PackNetLinear.push_pruned()`


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

    weight: Tensor
    """Weights of each connection"""
    bias: Tensor
    """Biases for each neuron"""
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

    in_features: int
    out_features: int
    weight_count: int

    @property
    def top_mask(self) -> Tensor:
        """Return a mask of weights at the top of the `stack`"""
        return self.z_mask.eq(self._z_top)

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.z_mask.eq(self._Z_PRUNED)

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_count = in_features * out_features

        # Initialize stack z index
        self.register_buffer("z_mask", torch.ones((out_features, in_features)).byte() * self._z_top)

        # Initialize tunable parameters
        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features)))
        self.bias = nn.parameter.Parameter(
            torch.empty(out_features)
        )

        self._reset_parameters()
        self.weight.register_hook(self._backward_hook)


    def _reset_parameters(self) -> None:
        # Kaiming initialization
        # Reused `nn.Linear` initialization code
        # See https://github.com/pytorch/pytorch/issues/57109 or
        # https://pouannes.github.io/blog/initialization/#xavier-and-kaiming-initialization
        # for details
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def _backward_hook(self, grad: Tensor) -> Tensor:
        """
        Only the top layer should be trained. Todo so all other gradients
        are zeroed. Caution should be taken when optimizers with momentum are 
        used, since they can cause parameters to be modified even when no 
        gradient exists
        """
        assert self._z_active == self._z_top, \
            "Backward not possible, only the top can be trained"
        grad[~self.top_mask] = 0.0
        return grad

    def _available_weights(self, z_index: int) -> Tensor:
        weight = self.weight.clone()
        # Mask of the weights that are above the supplied z_index. Used to zero
        # them 
        mask = self.z_mask.greater(z_index)
        with torch.no_grad(): weight[mask] = 0.0
        return weight

    def _prunable_count(self) -> int:
        return self.top_mask.count_nonzero()

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
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * to_prune_proportion)
        self._prune_weights(ranked[:prune_count])

    def use_task_subset(self, task_id):
        """Setter to set the sub-set of the network to be used on forward pass"""
        task_id = min(max(task_id, 0), self._z_top)
        self._z_active = task_id

    def use_top_subset(self):
        """Forward should use the top subset"""
        self.use_task_subset(self._z_top)

    def push_pruned(self):
        if self.pruned_mask.count_nonzero() == 0:
            raise RuntimeError(f"No pruned parameters exist to be pushed. Try pruning first. Weight count: {self.weight_count}")
        # The top is now one higher up
        self._z_top += 1
        # Move pruned weights to the top
        self.z_mask[self.pruned_mask] = self._z_top
        # Change the active z_index
        self.use_top_subset()
        
        """
        Freezing bias parameters
        "We did not find it necessary to learn task-specific biases similar to EWC,
        and keep the biases of all the layers fixed after the network is pruned and
        re-trained for the first time." (Mallya & Lazebnik, 2018)
        """
        self.bias.requires_grad = False

    def forward(self, input: Tensor) -> Tensor:
        weights = self._available_weights(self._z_active)
        return F.linear(input, weights, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
