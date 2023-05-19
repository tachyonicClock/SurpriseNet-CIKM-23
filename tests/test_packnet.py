from surprisenet.packnet import _PnLinear, _TaskMaskParent
from surprisenet.mask import StateError

import random
import numpy as np
import torch
import pytest


def seed_all():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_packnet():
    seed_all()
    linear = _PnLinear(torch.nn.Linear(10, 10))
    assert linear.weight.count_nonzero() == 100

    linear.forward(torch.randn(10)).sum().backward()
    assert linear.weight.grad.count_nonzero() == 100
    linear.weight.grad = None  # Zero out the gradient

    # Transition to PRUNED_TOP
    linear.prune(0.5)
    assert linear.available_weights().count_nonzero() == 50
    with pytest.raises(StateError):
        linear.prune(0.5)

    linear(torch.randn(10)).sum().backward()
    remaining_grad = linear.weight.grad
    assert remaining_grad.count_nonzero() == 50
    linear.weight.grad = None  # Zero out the gradient

    # Transition to MUTABLE_TOP
    linear.push_pruned()
    assert linear.weight.count_nonzero() == 100
    linear.mutable_activate_subsets([0])

    linear(torch.randn(10)).sum().backward()
    pruned_grad = linear.weight.grad
    assert pruned_grad.count_nonzero() == 50
    linear.weight.grad = None  # Zero out the gradient

    assert (pruned_grad + remaining_grad).count_nonzero() == 100

    linear.activate_subsets([0])
    print(linear.visiblity_mask)
    assert linear.available_weights().count_nonzero() == 50


class _SampleParent(_TaskMaskParent):
    def __init__(self):
        super().__init__()
        self.linear = _PnLinear(torch.nn.Linear(10, 10))

    def forward(self, x):
        return self.linear(x)


def test_packnet_parent():
    model = _SampleParent()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    def train():
        for _ in range(10):
            optim.zero_grad()
            model(torch.randn(10)).sum().backward()
            optim.step()

    train()
    model.prune(0.5)
    model.push_pruned()

    model.activate_subsets([0])
    non_zero = model.linear.available_weights().not_equal(0)

    model.mutable_activate_subsets([])
    non_zero_b = model.linear.available_weights().not_equal(0)

    assert (non_zero ^ non_zero_b).all()
