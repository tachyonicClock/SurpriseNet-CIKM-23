

import numpy as np
from torchmetrics import Accuracy

from avalanche.evaluation.metrics.loss import LossPluginMetric
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue


class _MyMetric(PluginMetric[float]):

    def before_training_exp(self, strategy):
        self.strategy = strategy

    @property
    def exp(self) -> int:
        return self.strategy.clock.train_exp_counter

class EpochClock(_MyMetric):

    def after_training_epoch(self, strategy) -> MetricValue:
        clock = strategy.clock
        epoch = clock.train_exp_epochs
        step = strategy.clock.total_iterations
        return MetricValue(self, f"clock/{self.exp:04d}_epoch", epoch, step)

    def result(self, **kwargs):
        return super().result(**kwargs)

    def reset(self, **kwargs) -> None:
        return super().reset(**kwargs)


# class ReconLossMB(_MyMetric):

#     def after_eval_iteration(self, strategy):
#         return super().after_eval_iteration(strategy)