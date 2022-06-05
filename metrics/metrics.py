

import logging
import pickle
import typing as t
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from sympy import ones

from torch import Tensor
import torch
from torchmetrics import ConfusionMatrix
from avalanche.evaluation.metrics.loss import LossPluginMetric
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_definitions import Metric

from avalanche.evaluation.metric_results import MetricValue
from experiment.loss import LossObjective
from experiment.strategy import Strategy
from functional import figure_to_image


class _MyMetric(PluginMetric[float]):

    def before_training_exp(self, strategy):
        self.strategy = strategy

    @property
    def train_experience(self) -> int:
        """
        The experience that is currently being trained on. Or was previously 
        trained on if we are in the eval phase
        """
        return self.strategy.clock.train_exp_counter

    @property
    def test_experience(self) -> int:
        """
        The experience we are currently being tested on
        """
        return self.strategy.experience.current_experience

    def reset(self):
        pass
    def result(self):
        pass

class EpochClock(_MyMetric):

    def after_training_epoch(self, strategy) -> MetricValue:
        clock = strategy.clock
        epoch = clock.train_exp_epochs
        step = strategy.clock.total_iterations
        return MetricValue(self, f"clock/{self.train_experience:04d}_epoch", epoch, step)

    def result(self, **kwargs):
        return super().result(**kwargs)

    def reset(self, **kwargs) -> None:
        return super().reset(**kwargs)


class ExperienceIdentificationCM(_MyMetric):

    cm: ConfusionMatrix

    def __init__(self, n_experiences: int) -> None:
        self.n_experiences = n_experiences
        self.cm = ConfusionMatrix(n_experiences)
        self.reset()
        print("ExperienceIdentificationCM")

    def update(self, preds: Tensor, target: int):
        self.cm.update(preds.cpu(), torch.ones((preds.shape)).int() * target)

    def result(self):
        cm = self.cm.compute()
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_ylabel("True Experience")
        ax.set_xlabel("Predicted Experience")
        return figure_to_image(fig)

    def reset(self):
        self.cm.reset()

    def before_eval(self, strategy):
        self.reset()

    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        exp_id = strategy.experience.current_experience
        pred_exp_id = strategy.last_forward_output.pred_exp_id
        assert pred_exp_id != None, "Strategy did not output pred_exp_id"
        self.update(pred_exp_id, exp_id)
    
    def after_eval(self, strategy: Strategy):
        return MetricValue(self, f"ExperienceIdentificationCM", self.result(), strategy.clock.total_iterations)


class ConditionalMetrics(_MyMetric):

    correct_and_correct_task_id: int
    correct_task_id: int
    correct_and_wrong_task_id: int
    wrong_task_id: int

    def __init__(self):
        self.reset()

    @property
    def correct_given_correct_task_id(self) -> float:
        """P(correct|correct exp id)"""
        return float(self.correct_and_correct_task_id/self.correct_task_id)

    @property
    def correct_given_wrong_task_id(self) -> float:
        """P(correct|wrong exp id)"""
        return float(self.correct_and_wrong_task_id/self.wrong_task_id)

    @property
    def task_id_accuracy(self) -> float:
        """P(correct_exp_id)"""
        return float(self.correct_task_id / (self.wrong_task_id + self.correct_task_id))


    def update(self, y:Tensor, y_hat:Tensor, task_label:Tensor, task_pred:Tensor):

        correct_class = y_hat.eq(y)
        correct_task  = task_pred.eq(task_label)

        self.correct_and_correct_task_id += \
             (correct_class * correct_task).count_nonzero()

        self.correct_and_wrong_task_id += \
             (correct_class * ~correct_task).count_nonzero()

        self.correct_task_id += correct_task.count_nonzero()
        self.wrong_task_id += (~correct_task).count_nonzero()


    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        exp_id = strategy.experience.current_experience
        if exp_id >= strategy.clock.train_exp_counter:
            return
        out = strategy.last_forward_output
        y_hat = out.y_hat.argmax(dim=1).cpu()
        task_label = exp_id * torch.ones(y_hat.size()).int()
        self.update(strategy.mb_y.cpu(), y_hat, task_label, out.pred_exp_id.cpu())

    def before_eval(self, strategy):
        self.reset()

    def after_eval(self, strategy: Strategy):
        step = strategy.clock.total_iterations

        return [
            MetricValue(self, f"Conditional/P(correct|correct_task_id)", self.correct_given_correct_task_id, step),
            MetricValue(self, f"Conditional/P(correct|!correct_task_id)", self.correct_given_wrong_task_id, step),
            MetricValue(self, f"Conditional/P(correct_task_id)", self.task_id_accuracy, step)
        ]

    def reset(self):
        self.correct_and_correct_task_id = 0
        self.correct_task_id = 0
        self.correct_and_wrong_task_id = 0
        self.wrong_task_id = 0

    def result(self, **kwargs):
        return None


        
        
class LossObjectiveMetric(_MyMetric):

    n_samples: int
    loss_sum: float

    def __init__(self, name: str, loss_part: LossObjective):
        super().__init__()
        self.loss_part = loss_part
        self.name = name
        self.reset()

    def after_training_iteration(self, strategy: Strategy) -> "MetricResult":
        self.loss_sum += float(self.loss_part.loss)
        self.n_samples += 1

    def reset(self):
        self.n_samples = 0.0
        self.loss_sum = 0.0

    def result(self):
        return self.loss_sum/self.n_samples

    def after_training_epoch(self, strategy: Strategy) -> "MetricResult":
        step = strategy.clock.total_iterations
        value = self.result()
        self.reset()
        return MetricValue(self, f"TrainLossPart/{self.name}", value, step)

class EvalLossObjectiveMetric(_MyMetric):

    n_samples: int
    loss_sum: float

    def __init__(self, name: str, loss_part: LossObjective):
        super().__init__()
        self.loss_part = loss_part
        self.name = name
        self.reset()

    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        self.loss_sum += float(self.loss_part.loss)
        self.n_samples += 1

    def reset(self):
        self.n_samples = 0.0
        self.loss_sum = 0.0

    def result(self):
        return self.loss_sum/self.n_samples

    def after_eval_exp(self, strategy: Strategy) -> "MetricResult":
        step = strategy.clock.total_iterations
        value = self.result()
        self.reset()
        return MetricValue(self, f"EvalLossPart/Experience_{self.test_experience}/{self.name}", value, step)


class TaskInferenceMetrics(_MyMetric):
    def __init__(self, logdir: str):
        super().__init__()
        self.loss_points: t.Set[t.Tuple[int, int, float]] = set()
        self.logdir = logdir


    def add_point(self):
        pass


    def after_eval_iteration(self, strategy: Strategy) -> "MetricResult":
        out = strategy.last_forward_output
        lbl = out.loss_by_layer

        assert lbl != None, "Expected loss by layer to be populated"
        
        for layer in range(lbl.shape[0]):
            for instance in range(lbl.shape[1]):
                loss = lbl[layer, instance]
                self.loss_points.add((layer, self.test_experience, float(loss))) 


    def after_eval(self, strategy: Strategy) -> "MetricResult":

        # print(self.loss_points)
        with open(f"{self.logdir}/{self.train_experience}_loss_points.pickle", "wb") as f:
            pickle.dump(self.loss_points, f)

        self.loss_points = set()



