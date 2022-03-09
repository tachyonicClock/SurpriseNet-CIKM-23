import math
from unicodedata import decimal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class PackNetLinear(nn.Module):
    """
    Implementation of linear module to implement PackNet functionality. You
    can think about a PackNet as a Stack of networks overlayed on top of each other
    where each layer is only connected to those below additionally only the
    top of the stack can be modified.

    In order to grow the stack the top must be pruned to make way for it.


    Notes:
    "We did not find it necessary to learn task-specific biases similar to EWC,
    and keep the biases of all the layers fixed after the network is pruned and
    re-trained for the first time." (Mallya & Lazebnik, 2018)

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

    stack_z: Tensor
    """
    Stack z is a depth index of each neuron in an imaginary "stack"

    A stack consists of:
     * `z = -1` Represents pruned weights reserved for future use
     * `z = 0`  Represents the top of the stack where a weight can be trained
     * `z > 0`  Represents a weight within the stack that cannot be trained

    """

    in_features: int
    out_features: int
    weight_count: int

    weight: Tensor
    bias: Tensor
    

    def __init__(self, in_features: int, out_features: int, device=None) -> None:
        super().__init__()

        self.training_task = 0

        self.in_features = in_features
        self.out_features = out_features
        self.weight_count = in_features * out_features
        # Initialize stack z index
        self.stack_z = torch.zeros((out_features, in_features))

        # Initialize tunable parameters
        self.weight = nn.parameter.Parameter(
            torch.empty((out_features, in_features), device=device))
        self.bias = nn.parameter.Parameter(
            torch.empty(out_features, device=device)
        )

        self.reset_parameters()

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

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def _prunable_mask(self) -> Tensor:
        """Return a mask of prunable neurons"""
        return self.stack_z.eq(0)

    def _prunable_count(self) -> int:
        return self._prunable_mask().count_nonzero()

    def _rank_prunable(self) -> Tensor:

        prunable_mask = self._prunable_mask()
        prunable_count = self._prunable_count()

        # "We use the simple heuristic to quantify the importance of the 
        # weights using their absolute value." (Han et al., 2017)
        importance = self.weight.abs() * prunable_mask
        
        # Rank the importance
        rank = torch.argsort(importance.flatten(), descending=True)
        return rank[:prunable_count]

    def _prune_weights(self, indices: Tensor):
        self.stack_z.flatten()[indices] = -1.0
    
    # def _prune_percentage(self, ranked: Tensor) -> Tensor:

    #     self.weight.flatten().scatter(0, ranked, Tensor(-1))
    #     pass




    pass

