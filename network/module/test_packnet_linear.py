import math
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import pytest

from packnet_linear import * 

TORCH_SEED = torch.seed()

layer_shapes = [(2,3), (8, 4), (64, 128), (128, 64), (1024, 1024)]

@pytest.fixture(params=layer_shapes)
def packnet_linear(request):
    torch.manual_seed(TORCH_SEED)
    return PackNetLinear(*request.param)

@pytest.fixture(params=layer_shapes)
def packnet_linear_random_z(request):
    """Same as `packnet_linear` but the z index is randomized"""
    torch.manual_seed(TORCH_SEED)
    packnet_linear = PackNetLinear(*request.param)
    packnet_linear.stack_z = torch.randint(0, 2, packnet_linear.stack_z.shape)
    return packnet_linear

@pytest.fixture(params=layer_shapes)
def torch_linear(request):
    torch.manual_seed(TORCH_SEED)
    return nn.Linear(*request.param)


def test_init(packnet_linear):
    assert packnet_linear

def test_equivalence(packnet_linear: PackNetLinear, torch_linear: nn.Linear):
    """Test `PackNetLinear` and `nn.Linear` are equivalent after init"""

    # Test only applicable when the layers are the same shape
    if packnet_linear.weight.shape != torch_linear.weight.shape:
        return

    assert packnet_linear.weight.equal(torch_linear.weight), "Weights should match"
    assert packnet_linear.bias.equal(torch_linear.bias), "Weights should match"

    x = torch.rand((1, packnet_linear.in_features))
    assert torch_linear.forward(x).equal(packnet_linear.forward(x)), \
        "Forward should be equivalent"

@pytest.mark.parametrize('_', range(5))
def test__prunable_mask(_, packnet_linear: PackNetLinear):
    """Test that prunable mask works"""
    z_stack_shape = packnet_linear.stack_z.shape
    n_weights = packnet_linear.out_features * packnet_linear.in_features

    prunable = n_weights // 2
    not_prunable = n_weights - prunable

    packnet_linear.stack_z = \
        torch.cat((torch.zeros(prunable), torch.randint(1, 10, (not_prunable,)))) \
        .reshape(z_stack_shape)

    assert packnet_linear._prunable_mask().count_nonzero().sum() == prunable



def test__rank_prunable(packnet_linear_random_z: PackNetLinear):
    ranked = packnet_linear_random_z._rank_prunable()

    # Get value of weights in order, if and only if they are prunable
    ranked_weights = packnet_linear_random_z.weight.flatten()[ranked].abs()

    assert ranked_weights.equal(ranked_weights.sort(descending=True).values), "Expected to be sorted already"
    assert ranked_weights.size()[0] == packnet_linear_random_z._prunable_count(), "Got the wrong number of ranked parameter indices"


def test__prune_weights(packnet_linear: PackNetLinear):
    prune_count = packnet_linear.weight_count//3
    to_prune = list(range(packnet_linear.weight_count))
    random.shuffle(to_prune)
    to_prune = to_prune[:prune_count]
    packnet_linear._prune_weights(Tensor(to_prune).long())
    assert packnet_linear.stack_z.eq(-1).count_nonzero() == prune_count

