
import dataclasses
from datetime import datetime
import random
from typing import Any, Dict, Generic, Iterator, Sequence, Type, TypeVar
import numpy as np
import pandas as pd
from torch import nn
import json
from dataclasses import dataclass
import torch

from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario
from avalanche.evaluation.metrics.accuracy import accuracy_metrics
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.strategies.strategy_wrappers import Naive
from network.simple_network import SimpleDropoutMLP

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class Experiment():
    scenario:    NCScenario = None
    model:       nn.Module = None
    evaluator:   EvaluationPlugin = None
    strategy:    BaseStrategy = None
    logdir:      str
    scheduler = None

    def config_file(self):
        return f"{self.logdir}/config.json"

    def train(self):
        """Train and test the experiment"""
        train_stream = self.scenario.train_stream
        test_stream = self.scenario.test_stream

        results = []
        for i, experience in enumerate(train_stream):
            self.strategy.train(experience)
            result = self.strategy.eval(test_stream[:i+1])
            results.append(result)

            if self.scheduler != None:
                self.scheduler.step()
        
        return results

@dataclass
class ExperimentConfig:
    lr: float          = 0.001
    momentum: float    = 0.9
    train_mb_size: int = 500
    eval_mb_size: int  = 100
    train_epochs: int  = 1
    n_experiences: int = 5
    device: str = "cuda"
    logdir_root: str = "./data/"


class ExperimentTemplate():

    config: ExperimentConfig
    title: str

    def make_scenario(self) -> NCScenario:
        pass

    def make_lr_schedule(self, optimizer) -> None:
        return None

    def make_logdir(self) -> str:
        return  self.config.logdir_root + datetime.now().strftime('%m_%d_%Y_%H_%M_%S_') + self.title

    def make_evaluator(self, n_classes) -> EvaluationPlugin:
        return EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True, trained_experience=True),
            suppress_warnings=True)
    
    def make_model(self, n_classes) -> nn.Module:
        return SimpleDropoutMLP(num_classes=n_classes)

    def make_optimizer(self, parameters: Iterator[Parameter]):
        return SGD(parameters, lr=self.config.lr, momentum=self.config.momentum)
    
    def make_strategy(self, model: nn.Module, optimizer: Optimizer, evaluator: EvaluationPlugin) -> BaseStrategy:
        return Naive(
            model, optimizer=optimizer, evaluator=evaluator,
            train_mb_size=self.config.train_mb_size,
            eval_mb_size=self.config.eval_mb_size,
            train_epochs=self.config.train_epochs,
            device=self.config.device)

    def build(self) -> Experiment:
        exp = Experiment()
        exp.scenario  = self.make_scenario()
        exp.evaluator = self.make_evaluator(exp.scenario.n_classes)
        exp.model     = self.make_model(exp.scenario.n_classes)
        optimizer = self.make_optimizer(exp.model.parameters())
        exp.strategy  = self.make_strategy(exp.model, optimizer, exp.evaluator)
        exp.scheduler = self.make_lr_schedule(optimizer)
        return exp

    def __init__(self, title, config=ExperimentConfig) -> None:
        self.config = config
        self.title  = title

class FashionExperiment(ExperimentTemplate):
    def make_scenario(self) -> NCScenario:
        return SplitFMNIST(n_experiences=self.config.n_experiences)

    




