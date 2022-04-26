import json
from config import get_logger
log = get_logger(__name__)

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
from torch import Tensor, nn
from torch.utils.tensorboard.summary import hparams
from experiment.loss import MultipleObjectiveLoss
from experiment.strategy import ForwardOutput, Strategy

from metrics.metrics import ConditionalMetrics, EpochClock, ExperienceIdentificationCM, LossObjectiveMetric

from config import *
from metrics.reconstructions import GenerateReconstruction, GenerateSamples

import setproctitle
from network.trait import AutoEncoder, ConditionedSample, InferTask, PackNet, Samplable, NETWORK_TRAITS



@dataclass
class BaseHyperParameters():
    lr: float
    
    train_mb_size: int
    eval_mb_size: int

    train_epochs: int
    eval_every: int
    device: str


class Experiment(SupervisedPlugin):
    """
    Py-lightning style container for continual learning
    """

    strategy: Strategy
    network:  nn.Module
    logger:   av.logging.TensorboardLogger
    scenario: av.benchmarks.NCScenario
    optimizer: torch.optim.Optimizer
    evaluator: EvaluationPlugin
    hp: BaseHyperParameters
    objective: MultipleObjectiveLoss

    def __init__(self, hp: BaseHyperParameters) -> None:
        super().__init__()

        self.hp = hp

        self.label = f"experiment_{max(self._get_log_numbers())+1:04d}"

        setproctitle.setproctitle(self.label)

        # Create a new logger with sequential names
        self.logdir = LOGDIR+"/"+self.label
        self.logger = av.logging.TensorboardLogger(self.logdir)

        self.objective = self.make_objective()
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
        exp, ssi, sei = hparams(hparam_dict, dependent_var)
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
        def _loss_function(output: Tensor, target: Tensor) -> Tensor:
            self.objective.update(self.last_mb_output, target)
            return self.objective.weighted_sum
        return _loss_function

    def make_objective(self) -> MultipleObjectiveLoss:
        return NotImplemented

    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        """Overload to define the evaluation plugin"""
        plugins = []
        if isinstance(self.network, AutoEncoder):
            plugins.append(GenerateReconstruction(self.scenario, 2, 1))
        if isinstance(self.network, Samplable):
            plugins.append(GenerateSamples(5, 4, rows_are_experiences=isinstance(self.network, ConditionedSample)))
        if isinstance(self.network, InferTask):
            plugins.append(ConditionalMetrics())
            plugins.append(ExperienceIdentificationCM(self.n_experiences))

        for name, objective in self.objective:
            plugins.append(LossObjectiveMetric(name, objective))

        return EvaluationPlugin(
            loss_metrics(epoch=True,
                         epoch_running=True, experience=True, stream=True, minibatch=True),
            accuracy_metrics(epoch=True, stream=True,
                             experience=True, trained_experience=True),
            confusion_matrix_metrics(num_classes=num_classes, stream=True),
            forgetting_metrics(experience=True, stream=True),
            EpochClock(),
            *plugins,
            loggers=loggers,
            suppress_warnings=True
        )

    def preflight(self):
        log.info(f"Network: {type(self.network)}")
        log.info(f"Traits:")
        for trait in NETWORK_TRAITS:
            if isinstance(self.network, trait):
                log.info(f" > Has the `{trait.__name__}` trait")
        log.info(f"Objectives:")
        for name, _ in self.objective:
            log.info(f" > Has the `{name}` objective")
        
        with open(self.logdir + "/hp.json", "w") as f:
            json.dump(self.hp.__dict__, f)

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

    def make_scenario(self) -> av.benchmarks.NCScenario:
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
    def n_classes(self) -> int:
        return self.scenario.n_classes

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
