from surprisenet.packnet import _PnLinear
from surprisenet.surprisenet_core import StateError

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

    linear(torch.randn(10)).sum().backward()
    pruned_grad = linear.weight.grad
    assert pruned_grad.count_nonzero() == 50
    linear.weight.grad = None  # Zero out the gradient

    assert (pruned_grad + remaining_grad).count_nonzero() == 100

    linear.use_task_subset(0)
    print(linear.visiblity_mask)
    assert linear.available_weights().count_nonzero() == 50
