import typing
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import functional as F

from experiment.strategy import ForwardOutput

class LossObjective(ABC):
    name: str = "Untitled"
    loss: Tensor = 0.0
    weighting: float = 1.0

    @property
    def weighted(self):
        return self.weighting * self.loss

    @abstractmethod
    def update(self, out: ForwardOutput, target: Tensor = None):
        pass

    def __init__(self, weighting: float = 1.0) -> None:
        self.weighting = weighting


class MultipleObjectiveLoss():
    objectives: typing.Dict[str, LossObjective]

    def __init__(self):
        self.objectives = dict()

    def add(self, objective: LossObjective) -> 'MultipleObjectiveLoss':
        self.objectives[objective.name] = objective
        return self

    def update(self, out: ForwardOutput, target: Tensor):
        assert isinstance(out, ForwardOutput), "Expected forward output"
        for _, objective in self.objectives.items():
            objective.update(out, target)

    def __iter__(self):
        return iter(self.objectives.items())

    @property
    def sum(self):
        sum = 0.0
        for _, objective in self:
            sum += 1/(len(self.objectives)) * objective.weighted
        return sum


class BCEReconstructionLoss(LossObjective):
    name = "BCEReconstruction"

    def __init__(self, weighting: float = 1) -> None:
        super().__init__(weighting)

    def update(self, out: ForwardOutput, target: Tensor = None):
        assert out.x.max() <= 1.0, "Input is not normalized"
        assert out.x.min() >= 0.0, "Input is not normalized"
        x_hat = out.x_hat.clamp(0, 1)
        self.loss = F.binary_cross_entropy(x_hat, out.x)

class MSEReconstructionLoss(LossObjective):
    name = "MSEReconstruction"

    def __init__(self, weighting: float = 1) -> None:
        super().__init__(weighting)

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = F.mse_loss(out.x_hat, out.x)

class ClassifierLoss(LossObjective):
    name = "Classifier"

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = F.cross_entropy(out.y_hat, target)

class VAELoss(LossObjective):
    name = "VAE"

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = torch.mean(-0.5 * torch.sum(1 + out.log_var - out.mu ** 2 - out.log_var.exp(), dim = 1), dim = 0)

