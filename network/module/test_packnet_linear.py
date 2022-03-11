import math
import random
from turtle import forward

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
    sorted_weights = packnet_linear_random_z.weight.flatten().abs()[ranked]
    double_sort = sorted_weights.sort().values

    # assert double_sort.indices.equal(ranked)

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

    # Get the smallest parameter that should definitely be pruned
    smallest = packnet_linear.weight.flatten().abs().min()

    packnet_linear.prune(proportion)
    assert packnet_linear.pruned_mask.count_nonzero() == should_prune

    # Get the new smallest parameter from the top
    new_smallest = packnet_linear.weight[packnet_linear.top_mask].abs().min()

    packnet_linear.push_pruned()
    assert packnet_linear.top_mask.count_nonzero() == should_prune
    assert packnet_linear.pruned_mask.count_nonzero() == 0


    assert smallest < new_smallest, "Prune should remove the smallest absolute value"



# def XORNet(PackNet)
class XORNet(PackNetModule):
    def __init__(self) -> None:
        super().__init__()
        self.lin_1 = PackNetLinear(4, 8)
        self.lin_2 = PackNetLinear(8, 2)

    def forward(self, input: Tensor):
        x = self.lin_1(input)
        x = F.relu(x)
        x = self.lin_2(x)
        return x


def test_toy_end_to_end():
    """Learn task incremental XOR with PackNet"""

    task_a = (
        Tensor([[1, 0, 0, 0],[0, 1, 0, 0],[1, 1, 0, 0],[0, 0, 0, 0]]),
        Tensor([[1, 0],[1, 0],[0, 0],[0, 0]])
    )

    task_b = (
        Tensor([[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 1, 1],[0, 0, 0, 0]]),
        Tensor([[0, 1],[0, 1],[0, 0],[0, 0]])
    )

    model = XORNet()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = 2*1000
    prune_proportion = 0.75

    @torch.no_grad()
    def validate(step, dataset):
        x, y = dataset
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y)
        print(f"Step: {step:04d}, Loss: {float(loss):.5f}")
        return loss

    def run_experience(dataset):
        for step in range(steps):
            optimizer.zero_grad()

            x, y = dataset
            i = random.choices(range(len(x)), k=1)
            x = x[i]
            y = y[i]

            y_hat = model(x)

            loss = loss_func.forward(y_hat, y)
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                validate(step, dataset)
        return validate(step, dataset)

    lin = model.lin_1
    def abs_weight():
        return lin.weight.abs()


    print("Learning task a")
    before = abs_weight()
    run_experience(task_a)
    pre_prune = abs_weight()
    assert not before.equal(pre_prune), "Expected model to learn"
    print("A: pre-prune:\n", abs_weight(), "\n")

    model.prune(prune_proportion)
    print("A: post-prune:\n", abs_weight(), "\n")
    post_prune = abs_weight()
    assert not post_prune.equal(pre_prune), "Pruning changed nothing"

    a_relearn_loss = run_experience(task_a)
    print("A: re-learn:\n", abs_weight(), "\n")
    relearn = abs_weight()
    assert not relearn.equal(post_prune), "Failed to relearn anything"
    assert relearn[lin.pruned_mask].eq(0.0).all(), "Pruned weights should still be zero"

    model.push_pruned()
    frozen_bias = lin.bias

    print("Learning task B")
    b_loss = run_experience(task_b)
    print("B: pre-prune:\n", abs_weight(), "\n")
    assert not relearn.equal(abs_weight()), "Did not learn new task"
    assert lin.bias.equal(frozen_bias), "Bias should not have changed"

    frozen = ~lin.top_mask
    assert relearn[frozen].equal(abs_weight()[frozen]), "Frozen parameters should not have changed"
    
    print("Task a")
    # By setting the task id we recreate the conditions after it relearns
    model.set_task_id(0)
    assert a_relearn_loss == validate(0, task_a), "Should perform identical to before task_b was trained"

    model.reset_task_id()
    print("Task b")
    assert validate(0, task_b) == b_loss, "Should not have changed performance"
