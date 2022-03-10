import math

import torch
import torch.nn as nn
from torch import ByteTensor, Tensor
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

    z_mask: Tensor
    """
    Z mask is a depth index of each neuron in an imaginary "stack" which makes
    up the PackNet

    A stack consists of:
     * `z = 0`  Represents pruned weights reserved for future use
     * `z = 1`  Represents the top of the stack where a weight can be trained
     * `z > 1`  Represents a weight within the stack that cannot be trained
    """
    Z_PRUNED = 0
    Z_TOP = 1

    @property
    def top_mask(self) -> Tensor:
        """Return a mask of weights at the top of the `stack`"""
        return self.z_mask.eq(self.Z_TOP)

    @property
    def pruned_mask(self) -> Tensor:
        """Return a mask of weights that have been pruned"""
        return self.z_mask.eq(self.Z_PRUNED)


    in_features: int
    out_features: int
    weight_count: int

    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_count = in_features * out_features

        # Initialize stack z index
        self.register_buffer("z_mask", torch.ones((out_features, in_features)).byte() * self.Z_TOP)

        # Initialize tunable parameters
        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features)))
        self.bias = nn.parameter.Parameter(
            torch.empty(out_features)
        )

        self.reset_parameters()
        self.weight.register_hook(self._backward_hook)

    def reset_parameters(self) -> None:
        # Kaiming initialization
        # Reused `nn.Linear` initialization code
        # See https://github.com/pytorch/pytorch/issues/57109 or
        # https://pouannes.github.io/blog/initialization/#xavier-and-kaiming-initialization
        # for details
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, z_index=1) -> Tensor:
        weights = self._available_weights(z_index)
        return F.linear(input, weights, self.bias)

    def _backward_hook(self, grad: Tensor) -> Tensor:
        """
        Only the top layer should be trained. Todo so all other gradients
        are zeroed. Caution should be taken when optimizers with momentum are 
        used, since they can cause parameters to be modified even when no 
        gradient exists
        """
        grad[~self.top_mask] = 0.0
        return grad

    def _available_weights(self, z_index: int) -> Tensor:
        weight = self.weight
        # Mask of the weights that are above the supplied z_index. Used to zero
        # them 
        mask = self.z_mask.less(z_index)
        with torch.no_grad(): weight[mask] = 0.0
        return weight

    def _prunable_count(self) -> int:
        return self.top_mask.count_nonzero()

    def _rank_prunable(self) -> Tensor:

        prunable_mask = self.top_mask
        prunable_count = self._prunable_count()

        # "We use the simple heuristic to quantify the importance of the 
        # weights using their absolute value." (Han et al., 2017)
        importance = self.weight.abs() * prunable_mask
        
        # Rank the importance
        rank = torch.argsort(importance.flatten(), descending=True)
        return rank[:prunable_count]

    def _prune_weights(self, indices: Tensor):
        self.z_mask.flatten()[indices] = 0
        with torch.no_grad(): self.weight.flatten()[indices] = 0.0

    def prune(self, to_prune_proportion: float):
        print("PRUNE!")
        ranked = self._rank_prunable()
        prune_count = int(len(ranked) * to_prune_proportion)
        self._prune_weights(ranked[:prune_count])

    def push_pruned(self):
        if self.pruned_mask.count_nonzero() == 0:
            raise RuntimeError("No pruned parameters exist to be pushed. Try pruning first")
        self.z_mask += 1
        
        """
        Freezing bias parameters
        "We did not find it necessary to learn task-specific biases similar to EWC,
        and keep the biases of all the layers fixed after the network is pruned and
        re-trained for the first time." (Mallya & Lazebnik, 2018)
        """
        self.bias.requires_grad = False



