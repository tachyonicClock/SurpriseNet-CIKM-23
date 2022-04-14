from experiment.strategy import ForwardOutput
import typing
import torch
from torch.nn import functional as F
from torch import Tensor
from abc import ABC, abstractmethod

from functional import MRAE

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
        for _, objective in self:
            objective.update(out, target)

    def __iter__(self):
        return iter(self.objectives.items())

    @property
    def weighted_sum(self):
        weighted_sum = 0.0
        for _, objective in self:
            weighted_sum += objective.weighted
        return weighted_sum


class ReconstructionError(LossObjective):
    name = "Reconstruction"

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = MRAE(out.x_hat, out.x)

class ClassifierLoss(LossObjective):
    name = "Classifier"

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = F.cross_entropy(out.y_hat, target)

class VAELoss(LossObjective):
    name = "VAE"

    def __init__(self, M: int, N: int, beta: float):
        """Create a VAE loss function

        :param M: M is the size of the latent space
        :param N: N is the input size
        :param beta: The beta factor, typically between 0.001 and 10 (https://openreview.net/pdf?id=Sy2fzU9gl)
        """
        self.weighting = beta*M/N

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = torch.mean(-0.5 * torch.sum(1+out.log_var - out.mu.square() - out.log_var.exp()))
