from experiment.strategy import ForwardOutput
import typing
import torch
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
    name = "ReconstructionError"
    
    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = MRAE(out.x_hat, out.x)
