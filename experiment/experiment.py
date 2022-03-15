from lib2to3.pytree import Base
import os
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import avalanche as av
from avalanche.evaluation.metrics import (
    loss_metrics,
    forgetting_metrics,
    confusion_matrix_metrics,
    accuracy_metrics)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
import torch
from torch import nn
from torch.utils.tensorboard.summary import hparams

from metrics.featuremap import FeatureMap
from metrics.metrics import TrainExperienceLoss

from conf import *
from metrics.reconstructions import GenerateReconstruction, GenerateSamples
from network.trait import Generative, SpecialLoss, TaskAware, TraitPlugin


@dataclass
class BaseHyperParameters():
    lr: float
    train_mb_size: int
    train_epochs: int
    eval_mb_size: int
    eval_every: int
    device: str


class Experiment(SupervisedPlugin):
    """
    Py-lightning style container for continual learning
    """

    strategy: SupervisedTemplate
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

    def add_plugins(self) -> Sequence[BasePlugin]:
        """
        Overload to define a sequence of plugins that will be added to the 
        strategy
        """
        return []

    def _experience_log(self, exp: av.benchmarks.NCExperience):
        print(f"Start of experience: {exp.current_experience}")
        print(f"Current Classes:     {exp.classes_in_this_experience}")
        print(f"Experience size:     {len(exp.dataset)}")

    def train_experience(self, experience: av.benchmarks.NCExperience):
        self.strategy.train(experience)

    def train(self):
        self.preflight()
        results = []
        for exp in self.scenario.train_stream:
            self._experience_log(exp)
            self.train_experience(exp)
            test_subset = self.scenario.test_stream
            results.append(self.strategy.eval(test_subset))
        self.logger.writer.flush()
        return results

    def make_strategy_type(self):
        return SupervisedTemplate

    def make_strategy(self) -> SupervisedPlugin:
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

        generative = [
            GenerateReconstruction(self.scenario, 2, 1),
            # GenerateSamples(2, 4, img_size=2.0)
        ] if isinstance(self.network, Generative) else []

        return EvaluationPlugin(
            loss_metrics(epoch=True,
                         epoch_running=True, experience=True, stream=True, minibatch=True),
            accuracy_metrics(epoch=True, stream=True,
                             experience=True, trained_experience=True),
            confusion_matrix_metrics(num_classes=num_classes, stream=True),
            forgetting_metrics(experience=True, stream=True),
            TrainExperienceLoss(),
            FeatureMap(),
            *generative,
            loggers=loggers,
            suppress_warnings=True
        )

    def preflight(self):
        print(f"Network: {type(self.network)}")
        for trait in [Generative, TaskAware, SpecialLoss]:
            if isinstance(self.network, trait):
                print(f" * Has the {trait} trait")

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
            name,
            value,
            step if step else self.strategy.clock.total_iterations)

    @property
    def lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def n_experiences(self) -> int:
        return len(self.scenario.train_stream)

    @property
    def clock(self):
        return self.strategy.clock

    def _get_log_numbers(self):
        for filename in os.listdir(LOGDIR):
            name, _ = os.path.splitext(filename)
            yield int(name[-4:])
        yield 0
