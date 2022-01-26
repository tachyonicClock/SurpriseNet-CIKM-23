import os
from typing import Sequence

import avalanche as av
import torch
from torch import nn
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch.utils.tensorboard.summary import hparams
from conf import *

class Experiment(StrategyPlugin):
    strategy: BaseStrategy
    network:  nn.Module
    logger:   av.logging.TensorboardLogger
    scenario: av.benchmarks.ScenarioStream
    optimizer: torch.optim.Optimizer
    parameters: dict = {}

    def _get_log_numbers(self):
        for filename in os.listdir(LOGDIR):
            name, _ = os.path.splitext(filename)
            yield int(name[-4:])
        yield 0

    def __init__(self) -> None:
        super().__init__()

        # Create a new logger with sequential names
        self.logger = av.logging.TensorboardLogger(
            LOGDIR+f"/experiment_{max(self._get_log_numbers())+1:04d}")

        self.scenario = self.make_scenario()
        self.network  = self.make_network()
        evaluator     = self.make_evaluator([self.logger, InteractiveLogger()], self.scenario.n_classes)
        self.optimizer     = self.make_optimizer(self.network.parameters())

        self.strategy = BaseStrategy(
            self.network,
            self.optimizer,
            **self.log_hparam(**self.configure_regime()),
            device="cuda",
            plugins=[self, *self.add_plugins()],
            evaluator=evaluator
        )

        exp, ssi, sei = hparams(self.parameters, {
            "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000": 0.0,
            "StreamForgetting/eval_phase/test_stream": 0.0
            })
        self.logger.writer.file_writer.add_summary(exp)
        self.logger.writer.file_writer.add_summary(ssi)
        self.logger.writer.file_writer.add_summary(sei)

    def add_plugins(self) -> Sequence[StrategyPlugin]:
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

    def log_hparam(self, **kwargs) -> dict:
        """Log a hyper parameter"""
        self.parameters.update(kwargs)
        return kwargs

    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        raise NotImplemented

    def make_network(self) -> nn.Module:
        raise NotImplemented

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        raise NotImplemented

    def make_scenario(self) -> av.benchmarks.ScenarioStream:
        raise NotImplemented

    def configure_regime(self) -> dict:
        return {}

    def log_scalar(self, name, value, step=None):
        self.logger.log_single_metric(name, value, step if step else self.strategy.clock.total_iterations)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']