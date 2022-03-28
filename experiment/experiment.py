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
from experiment.strategy import ForwardOutput, Strategy

from metrics.metrics import ConditionalMetrics, EpochClock, ExperienceIdentificationCM, LossPartMetric

from conf import *
from metrics.reconstructions import GenerateReconstruction, GenerateSamples
from network.deep_generative import Loss

from network.trait import AutoEncoder, PackNet, Samplable, get_all_trait_types

# Setup logging
import logging
import coloredlogs
coloredlogs.install(fmt='%(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

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

    strategy: Strategy
    network:  nn.Module
    logger:   av.logging.TensorboardLogger
    scenario: av.benchmarks.ScenarioStream
    optimizer: torch.optim.Optimizer
    evaluator: EvaluationPlugin
    hp: BaseHyperParameters
    loss: Loss

    def __init__(self, hp: BaseHyperParameters) -> None:
        super().__init__()

        logging.basicConfig(level=logging.INFO)

        self.hp = hp

        # Create a new logger with sequential names
        self.logger = av.logging.TensorboardLogger(
            LOGDIR+f"/experiment_{max(self._get_log_numbers())+1:04d}")

        self.loss = self.make_mulitpart_loss()
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
        log.info(f"Start of experience: {exp.current_experience}")
        log.info(f"Current Classes:     {exp.classes_in_this_experience}")
        log.info(f"Experience size:     {len(exp.dataset)}")

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
        return Strategy

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
            plugins=[self, *self.add_plugins()],
            evaluator=self.evaluator
        )

    def make_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def make_mulitpart_loss(self) -> Loss:
        return NotImplemented

    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        """Overload to define the evaluation plugin"""
        plugins = []
        if isinstance(self.network, AutoEncoder):
            plugins.append(GenerateReconstruction(self.scenario, 2, 1))
        if isinstance(self.network, Samplable):
            plugins.append(GenerateSamples(5, 4, rows_are_experiences=True))

        if self.loss.classifier.is_used:
            plugins.append(LossPartMetric("Classifier", self.loss.classifier))
        if self.loss.recon.is_used:
            plugins.append(LossPartMetric("Reconstruction", self.loss.recon))

        return EvaluationPlugin(
            loss_metrics(epoch=True,
                         epoch_running=True, experience=True, stream=True, minibatch=True),
            accuracy_metrics(epoch=True, stream=True,
                             experience=True, trained_experience=True),
            confusion_matrix_metrics(num_classes=num_classes, stream=True),
            forgetting_metrics(experience=True, stream=True),
            ConditionalMetrics(),
            ExperienceIdentificationCM(self.n_experiences),
            EpochClock(),
            *plugins,
            loggers=loggers,
            suppress_warnings=True
        )

    def preflight(self):
        log.info(f"Network: {type(self.network)}")
        for trait in get_all_trait_types():
            if isinstance(self.network, trait):
                log.info(f" > Has the {trait} trait")

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
        self.logger.writer.add_scalar(
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
    def last_mb_output(self) -> ForwardOutput:
        return self.strategy.last_forward_output

    @property
    def clock(self):
        return self.strategy.clock

    def _get_log_numbers(self):
        for filename in os.listdir(LOGDIR):
            name, _ = os.path.splitext(filename)
            yield int(name[-4:])
        yield 0
