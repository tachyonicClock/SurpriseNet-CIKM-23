import typing
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
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


class MultipleObjectiveLoss:
    objectives: typing.Dict[str, LossObjective]

    def __init__(self):
        self.objectives = dict()

    def add(self, objective: LossObjective) -> "MultipleObjectiveLoss":
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
            sum += objective.weighted
        return sum


class BCEReconstructionLoss(LossObjective):
    name = "BCEReconstruction"

    def __init__(self, weighting: float = 1) -> None:
        super().__init__(weighting)

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = F.binary_cross_entropy(out.x_hat, out.x)


class MSEReconstructionLoss(LossObjective):
    name = "MSEReconstruction"

    def __init__(self, weighting: float = 1) -> None:
        super().__init__(weighting)

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = F.mse_loss(out.x_hat, out.x)


class CrossEntropy(LossObjective):
    name = "Classifier"

    def update(self, out: ForwardOutput, target: Tensor = None):
        assert out.y_hat is not None, "Expected y_hat to be provided"
        # I ran into an issue when using the GenerativeReplay plugin
        # where the wrong type is used
        target = target.type(torch.LongTensor).to(out.y_hat.device)
        self.loss = F.cross_entropy(out.y_hat, target)


class ClassifierLossMasked(LossObjective):
    name = "ClassifierLossMasked"

    def update(self, out: ForwardOutput, target: Tensor = None):
        # Ignore the loss for classes that are not in the batch
        batch_classes = torch.unique(target)
        # Nothing to learn from classification if there is only one class
        if len(batch_classes) <= 1:
            self.loss = 0.0
            return

        batch_class_mask = torch.zeros_like(out.y_hat)
        for batch_class in batch_classes:
            batch_class_mask[:, batch_class] = 1.0

        out.y_hat = out.y_hat * batch_class_mask
        self.loss = F.cross_entropy(out.y_hat, target)


class VAELoss(LossObjective):
    name = "VAE"

    def update(self, out: ForwardOutput, target: Tensor = None):
        self.loss = torch.mean(
            -0.5 * torch.sum(1 + out.log_var - out.mu**2 - out.log_var.exp(), dim=1),
            dim=0,
        )


class _LogitNormLoss(nn.Module):
    def __init__(self, t=1.0):
        super(_LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


class LogitNorm(LossObjective):
    name = "LogitNorm"

    def __init__(self, weight: float, temperature: float = 1.0) -> None:
        super().__init__()
        self.loss_func = _LogitNormLoss(temperature)
        self.weighting = weight

    def update(self, out: ForwardOutput, target: Tensor = None):
        assert out.y_hat is not None, "Expected y_hat to be provided"
        self.loss = self.loss_func(out.y_hat, target)
