import os
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import avalanche as av
from avalanche.evaluation.metrics import loss_metrics, forgetting_metrics, confusion_matrix_metrics, accuracy_metrics
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy

import torch
from torch import nn
from torch.utils.tensorboard.summary import hparams
from experiment.experiment_strategy import ExperimentStrategy

from metrics.featuremap import FeatureMap
from metrics.metrics import TrainExperienceLoss

from conf import *
from metrics.reconstructions import GenerateReconstruction
from network.trait import TraitPlugin


@dataclass
class BaseHyperParameters():
    lr: float
    train_mb_size: int
    train_epochs: int
    eval_mb_size: int
    eval_every: int
    device: str


class Experiment(StrategyPlugin):
    """
    Py-lightning style container for continual learning
    """

    strategy: ExperimentStrategy
    network:  nn.Module
    logger:   av.logging.TensorboardLogger
    scenario: av.benchmarks.ScenarioStream
    optimizer: torch.optim.Optimizer
    evaluator: EvaluationPlugin
    hp: BaseHyperParameters

    def __init__(self, hp: BaseHyperParameters) -> None:
        super().__init__()
        self.hp = hp

        # Create a new logger with sequential names
        self.logger = av.logging.TensorboardLogger(
            LOGDIR+f"/experiment_{max(self._get_log_numbers())+1:04d}")

        self.scenario = self.make_scenario()
        self.network = self.make_network()
        self.evaluator = self.make_evaluator(
            [self.logger], self.scenario.n_classes)
        self.optimizer = self.make_optimizer(self.network.parameters())

        self.strategy = self.make_strategy()

        dependent_var = {}
        for x in self.make_dependent_variables():
            dependent_var[x] = 0.0

        # Turn hyper-parameters into a format that plays nice with tensorboard
        hparam_dict = {}
        discrete_hparam = {}
        for key, value in self.hp.__dict__.items():
            if isinstance(value, Enum):
                discrete_hparam[key] = [e.value for e in value]
            else:
                hparam_dict[key] = value

        exp, ssi, sei = hparams(hparam_dict, dependent_var, discrete_hparam)
        self.logger.writer.file_writer.add_summary(exp)
        self.logger.writer.file_writer.add_summary(ssi)
        self.logger.writer.file_writer.add_summary(sei)

    def add_plugins(self) -> Sequence[StrategyPlugin]:
        """
        Overload to define a sequence of plugins that will be added to the strategy
        """
        return []

    def train(self):
        results = []
        for i, experience in enumerate(self.scenario.train_stream):

            print("Start of experience: ", experience.current_experience)
            print("Current Classes:     ", experience.classes_in_this_experience)
            self.strategy.train(experience)
            test_subset = self.scenario.test_stream[:i+1]
            results.append(self.strategy.eval(test_subset))
        return results

    def make_strategy_type(self):
        return BaseStrategy

    def make_strategy(self) -> BaseStrategy:
        return self.make_strategy_type()(
            self.network,
            self.optimizer,
            criterion=self.make_criterion(),
            device=self.hp.device,
            train_mb_size=self.hp.train_mb_size,
            train_epochs=self.hp.train_epochs,
            eval_mb_size=self.hp.eval_mb_size,
            eval_every=self.hp.eval_every,
            plugins=[self, TraitPlugin(), *self.add_plugins()],
            evaluator=self.evaluator
        )

    def make_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        """Overload to define the evaluation plugin"""
        return EvaluationPlugin(
            loss_metrics(minibatch=True, epoch=True,
                         epoch_running=True, experience=True, stream=True),
            accuracy_metrics(epoch=True, stream=True,
                             experience=True, trained_experience=True),
            confusion_matrix_metrics(num_classes=num_classes, stream=True),
            forgetting_metrics(experience=True, stream=True),
            TrainExperienceLoss(),
            FeatureMap(),
            GenerateReconstruction(self.scenario, 1, 1),
            loggers=loggers,
            suppress_warnings=True
        )

    def make_network(self) -> nn.Module:
        raise NotImplemented

    def make_dependent_variables(self):
        return [
            "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000",
            "StreamForgetting/eval_phase/test_stream",
            "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000",
            "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001"
        ]

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        raise NotImplemented

    def make_scenario(self) -> av.benchmarks.ScenarioStream:
        raise NotImplemented

    def log_scalar(self, name, value, step=None):
        self.logger.log_single_metric(
            name, value, step if step else self.strategy.clock.total_iterations)

    @property
    def lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def n_experiences(self) -> int:
        return len(self.scenario.train_stream)

    def _get_log_numbers(self):
        for filename in os.listdir(LOGDIR):
            name, _ = os.path.splitext(filename)
            yield int(name[-4:])
        yield 0
