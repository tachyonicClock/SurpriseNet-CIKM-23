import os
import typing as t

import avalanche as av
from setproctitle import setproctitle
import torch
from avalanche.core import BasePlugin, SupervisedPlugin
from avalanche.evaluation.metrics import (accuracy_metrics,
                                          confusion_matrix_metrics,
                                          forgetting_metrics, loss_metrics)
from avalanche.training.plugins import EvaluationPlugin
from config.config import ExpConfig
from experiment.util import count_parameters
from metrics.metrics import (ConditionalMetrics, EpochClock,
                             EvalLossObjectiveMetric,
                             ExperienceIdentificationCM, LossObjectiveMetric)
from metrics.reconstructions import GenerateReconstruction, GenerateSamples
from network.trait import (NETWORK_TRAITS, AutoEncoder, Classifier,
                           ConditionedSample, InferTask, Samplable)
from torch import Tensor, nn
from torch.utils.tensorboard.summary import hparams

from experiment.loss import MultipleObjectiveLoss
from experiment.strategy import ForwardOutput, Strategy


class BaseExperiment(SupervisedPlugin):
    """
    Py-lightning inspired for continual learning with avalanche
    """

    strategy: Strategy
    network:  nn.Module
    logger:   av.logging.TensorboardLogger
    scenario: av.benchmarks.NCScenario
    optimizer: torch.optim.Optimizer
    evaluator: EvaluationPlugin
    objective: MultipleObjectiveLoss
    plugins: t.List[BasePlugin]
    cfg: ExpConfig

    def __init__(self, cfg: ExpConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.label = f"{max(self._get_log_numbers())+1:04d}_{self.cfg.name}"
        self.plugins = []
        setproctitle(self.label)

        # Create a new logger with sequential names
        self.logdir = self.cfg.tensorboard_dir+"/"+self.label
        self.logger = av.logging.TensorboardLogger(self.logdir)

        self.scenario = self.make_scenario()
        self.objective = self.make_objective()
        self.network = self.make_network()
        self.evaluator = self.make_evaluator(
            [self.logger], self.scenario.n_classes)
        self.optimizer = self.make_optimizer(self.network.parameters())

        self.strategy = self.make_strategy()

    def add_plugin(self, plugin: BasePlugin):
        self.plugins.append(plugin)

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

    def make_strategy(self) -> SupervisedPlugin:
        return NotImplemented

    def make_criterion(self):
        def _loss_function(output: Tensor, target: Tensor) -> Tensor:
            self.objective.update(self.last_mb_output, target)
            return self.objective.sum
        return _loss_function

    def make_objective(self) -> MultipleObjectiveLoss:
        return NotImplemented

    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        """Overload to define the evaluation plugin"""
        plugins = []
        is_images = self.cfg.is_image_data
        if isinstance(self.network, AutoEncoder) and is_images:
            plugins.append(GenerateReconstruction(self.scenario, 2, 1))
        if isinstance(self.network, Samplable) and is_images:
            plugins.append(GenerateSamples(
                5, 4, rows_are_experiences=isinstance(self.network, ConditionedSample)))

        if isinstance(self.network, InferTask):
            plugins.append(ConditionalMetrics())
            plugins.append(ExperienceIdentificationCM(self.n_experiences))

        if isinstance(self.network, Classifier):
            plugins.append(accuracy_metrics(epoch=True, stream=True,
                           experience=True, trained_experience=True))
            plugins.append(confusion_matrix_metrics(
                num_classes=num_classes, stream=True))
            plugins.append(forgetting_metrics(experience=True, stream=True))

        for name, objective in self.objective:
            plugins.append(LossObjectiveMetric(name, objective))
            plugins.append(EvalLossObjectiveMetric(name, objective))

        return EvaluationPlugin(
            loss_metrics(epoch=True, experience=True,
                         stream=True, minibatch=self.cfg.log_mini_batch),
            EpochClock(),
            *plugins,
            loggers=loggers
        )

    def dump_config(self):
        pass

    def preflight(self):
        print(f"Network: {type(self.network)}")
        print(f"Traits:")
        for trait in NETWORK_TRAITS:
            if isinstance(self.network, trait):
                print(f" > Has the `{trait.__name__}` trait")
        print(f"Objectives:")
        for name, _ in self.objective:
            print(f" > Has the `{name}` objective")

        print("-"*80)
        count_parameters(self.network)
        print("-"*80)
        print()
        self.dump_config()

    def make_network(self) -> nn.Module:
        raise NotImplemented

    def make_dependent_variables(self):
        return {
            "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000": 0,
            "Loss_Exp/eval_phase/test_stream/Task000/Exp000": 0,
            "EvalLossPart/Experience_0/Classifier": 0,
            "EvalLossPart/Experience_0/Reconstruction": 0
        }

    def add_hparams(self, hp: dict):
        print(hp)
        exp, ssi, sei = hparams(hp, self.make_dependent_variables(), {})
        self.logger.writer.file_writer.add_summary(exp)
        self.logger.writer.file_writer.add_summary(ssi)
        self.logger.writer.file_writer.add_summary(sei)

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
        for filename in os.listdir(self.cfg.tensorboard_dir):
            name, _ = os.path.splitext(filename)
            yield int(name[:4])
        yield 0
