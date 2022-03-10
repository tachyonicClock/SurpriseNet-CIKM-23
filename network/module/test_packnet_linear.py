import math
import random

import torch
import torch.nn as nn
from torch import Tensor, rand
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
    packnet_linear.z_mask = torch.randint(0, 2, packnet_linear.z_mask.shape)
    return packnet_linear

@pytest.fixture(params=layer_shapes)
def torch_linear(request):
    torch.manual_seed(TORCH_SEED)
    return nn.Linear(*request.param)


@pytest.fixture(params=["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return torch.device(request.param)


def test_init(packnet_linear: PackNetLinear, device):
    packnet_linear.to(device)
    assert packnet_linear.weight.device == device
    assert packnet_linear.bias.device == device
    assert packnet_linear.z_mask.device == device

def test_equivalence(packnet_linear: PackNetLinear, torch_linear: nn.Linear, device):
    """Test `PackNetLinear` and `nn.Linear` are equivalent after init"""
    packnet_linear.to(device)
    torch_linear.to(device)

    # Test only applicable when the layers are the same shape
    if packnet_linear.weight.shape != torch_linear.weight.shape:
        return

    assert packnet_linear.weight.equal(torch_linear.weight), "Weights should match"
    assert packnet_linear.bias.equal(torch_linear.bias), "Weights should match"

    x = torch.rand((1, packnet_linear.in_features)).to(device)
    assert torch_linear.forward(x).equal(packnet_linear.forward(x)), \
        "Forward should be equivalent"

def test__prunable_mask(packnet_linear: PackNetLinear, device):
    """Test that prunable mask works"""
    packnet_linear.to(device)
    z_stack_shape = packnet_linear.z_mask.shape
    n_weights = packnet_linear.out_features * packnet_linear.in_features

    prunable = n_weights // 2
    not_prunable = n_weights - prunable

    packnet_linear.z_mask = \
        torch.cat((torch.ones(prunable), torch.randint(2, 10, (not_prunable,)))) \
        .reshape(z_stack_shape).to(device)

    top_layer_count  = packnet_linear.top_mask.count_nonzero().sum()
    assert top_layer_count == prunable


def test__rank_prunable(packnet_linear_random_z: PackNetLinear, device):
    packnet_linear_random_z.to(device)
    ranked = packnet_linear_random_z._rank_prunable()

    # Get value of weights in order, if and only if they are prunable
    sorted_weights = packnet_linear_random_z.weight.flatten()[ranked].abs()
    double_sort = sorted_weights.sort(descending=True).values

    assert sorted_weights.equal(double_sort), "Expected to be sorted already"
    assert sorted_weights.size()[0] == packnet_linear_random_z._prunable_count(), "Got the wrong number of ranked parameter indices"


def test__prune_weights(packnet_linear: PackNetLinear, device):
    packnet_linear.to(device)
    prune_count = packnet_linear.weight_count//3
    to_prune = list(range(packnet_linear.weight_count))
    random.shuffle(to_prune)
    to_prune = to_prune[:prune_count]
    packnet_linear._prune_weights(Tensor(to_prune).long())

    assert packnet_linear.pruned_mask.count_nonzero() == prune_count
    assert packnet_linear.weight.eq(0.0).count_nonzero() == prune_count



def test__grad_freeze(packnet_linear_random_z, device):
    pnl = packnet_linear_random_z.to(device)
    x = pnl.forward(torch.ones(pnl.in_features).to(device))

    # Propagate gradients backwards. Gradients will be 1.0 except where
    # they are masked 
    x.sum().backward()
    grad = pnl.weight.grad.not_equal(0.0)
    assert pnl.top_mask.equal(grad), "Gradients should match mask "
    
@pytest.mark.parametrize('proportion', [0.1, 0.5, 0.75])
def test_prune(proportion, packnet_linear: PackNetLinear, device):
    packnet_linear.to(device)
    should_prune = int(packnet_linear.weight_count * proportion)
    if should_prune == 0:
        return

    packnet_linear.prune(proportion)
    assert packnet_linear.pruned_mask.count_nonzero() == should_prune

    packnet_linear.push_pruned()
    assert packnet_linear.top_mask.count_nonzero() == should_prune
    assert packnet_linear.pruned_mask.count_nonzero() == 0


