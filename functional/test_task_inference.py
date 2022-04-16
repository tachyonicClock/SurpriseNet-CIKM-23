import mock
import pytest

from torch import Tensor
import torch
from experiment.strategy import ForwardOutput, Network
from functional.task_inference import infer_task
from network.trait import AutoEncoder, PackNet


class MockAutoEncoder(AutoEncoder, PackNet, Network):

    active_task: int = 0

    def encode(self, x: Tensor) -> Tensor:
        pass

    def decode(self, z: Tensor) -> Tensor:
        pass

    def prune(self, to_prune_proportion: float) -> None:
        pass

    def push_pruned(self) -> None:
        pass

    @property
    def bottleneck_width(self) -> int:
        return 0

    def use_task_subset(self, task_id):
        self.active_task = task_id

    def use_top_subset(self):
        self.active_task = 0

    def forward(self, input: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        out.x = input
        out.x_hat = self.active_task * torch.ones_like(input)
        out.y_hat = torch.ones(input.shape[0]) * self.active_task
        return out

@pytest.fixture()
def very_simple_mock_data():
    x = [[0, 0], [1, 1], [2, 2], [1, 1]]
    y = [0, 1, 2, 1]
    return Tensor(x), Tensor(y)

@pytest.fixture()
def mock_ae():
    return MockAutoEncoder()


def test_mock_ae(very_simple_mock_data, mock_ae: MockAutoEncoder):
    x, y = very_simple_mock_data

    out_a = mock_ae.forward(x)
    assert out_a.x_hat.eq(0).all()

    mock_ae.use_task_subset(1)
    out_b = mock_ae.forward(x)
    assert out_b.x_hat.eq(1).all()


def test_task_infer(very_simple_mock_data, mock_ae: MockAutoEncoder):
    x, y = very_simple_mock_data

    out = infer_task(mock_ae, x, 3, 10)
    assert out.x_hat.equal(x)
    assert out.y_hat.equal(y)


