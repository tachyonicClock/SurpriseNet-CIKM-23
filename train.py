import os
import random
from typing import Sequence

import avalanche as av
import numpy as np

import torch
from torch import nn, Tensor

from avalanche.benchmarks.classic.cmnist import RotatedMNIST
from avalanche.evaluation.metrics.accuracy import accuracy_metrics
from avalanche.evaluation.metrics.confusion_matrix import \
    confusion_matrix_metrics
from avalanche.evaluation.metrics.forgetting_bwt import forgetting_metrics
from avalanche.evaluation.metrics.loss import LossPluginMetric, loss_metrics
from avalanche.training.plugins import ReplayPlugin, StrategyPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.storage_policy import (ClassBalancedBuffer,
                                               ExemplarsBuffer)

from conf import *
from experiment import Experiment
from metrics.metrics import TrainExperienceLoss
from module.dropout import ConditionedDropout

class ConditionedNetwork(nn.Module):

    def __init__(self, p_active, p_inactive, n_groups, layer_size) -> None:
        super().__init__()
        self.dropout_layers: Sequence[ConditionedDropout] = []

        def make_dropout(in_features):
            layer = ConditionedDropout(in_features, n_groups, p_active, p_inactive)
            self.dropout_layers.append(layer)
            return layer

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, layer_size),
            make_dropout(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            make_dropout(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 10),
        )

    def set_active_group(self, group_id):
        for layer in self.dropout_layers:
            layer.set_active_group(group_id)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        return self.layers(x)


dataset = av.benchmarks.classic.SplitFMNIST(
            dataset_root=DATASETS,
            n_experiences=2,
        )

class MyExperiment(Experiment):

    group: int = 0

    def __init__(self, hyper_params) -> None:
        self.p_active = hyper_params["p_active"]
        self.p_inactive = hyper_params["p_inactive"]
        self.hyper_params = hyper_params

        super().__init__()

    def make_network(self) -> nn.Module:
        return ConditionedNetwork(
            **self.log_hparam(p_inactive=self.p_inactive, p_active=self.p_active, layer_size=self.hyper_params["layer_size"]), 
            n_groups=self.scenario.n_classes)
    
    def make_evaluator(self, loggers, num_classes) -> EvaluationPlugin:
        return EvaluationPlugin(
            loss_metrics(minibatch=True, epoch=True, epoch_running=True, experience=True, stream=True),
            accuracy_metrics(epoch=True, stream=True, experience=True, trained_experience=True),
            confusion_matrix_metrics(num_classes=num_classes, stream=True),
            forgetting_metrics(experience=True, stream=True),
            TrainExperienceLoss(),
            loggers=loggers,
            suppress_warnings=True
        )

    def after_training_exp(self, _):
        self.group += 1
        self.network.set_active_group(self.group)

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self.schedule.step()
        self.log_scalar("lr", self.get_lr(), self.strategy.clock.total_iterations)

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(parameters, **self.log_hparam(lr=self.hyper_params["lr"]))
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer,  **self.log_hparam(gamma=self.hyper_params["exponential_decay"]))
        return optimizer

    def configure_regime(self) -> dict:
        return dict(train_mb_size = 100, train_epochs = self.hyper_params["epochs"], eval_mb_size = 1000)

    def make_scenario(self):
        return dataset


def random_params():
    return {
        "lr": 0.02,
        "epochs": 10,
        "exponential_decay": 0.7,
        "layer_size": 1024,
        "p_active": np.random.uniform(0.0, 1.0),
        "p_inactive": np.random.uniform(0.0, 1.0),
    }


for _ in range(1000):
    MyExperiment(random_params()).train()

