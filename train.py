import os
from typing import Sequence
from torch import nn, optim
import torch
from torch.functional import Tensor, norm
from torch.utils.tensorboard.summary import hparams
from torchvision import datasets
import avalanche as av
from avalanche.evaluation.metrics.confusion_matrix import confusion_matrix_metrics
from avalanche.evaluation.metrics.loss import LossPluginMetric, loss_metrics
from avalanche.evaluation.metrics.forgetting_bwt import forgetting_metrics
from avalanche.evaluation.metrics.accuracy import accuracy_metrics
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins import StrategyPlugin, ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer, ExemplarsBuffer

from avalanche.training.strategies.base_strategy import BaseStrategy
from metrics import TrainExperienceLoss
import numpy as np
import random

DATASETS = "./datasets"
LOGDIR = "./experiment_logs"

class Experiment(StrategyPlugin):
    strategy: BaseStrategy
    network:  nn.Module
    logger:   av.logging.TensorboardLogger
    scenario: av.benchmarks.ScenarioStream
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
        optimizer     = self.make_optimizer(self.network.parameters())

        self.strategy = BaseStrategy(
            self.network,
            optimizer,
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



class MyNetwork(nn.Module):

    def __init__(self, dropout_rate=0.0) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x: Tensor):
        x = self.flatten(x)
        return self.layers(x)


dataset = av.benchmarks.classic.cfashion_mnist.SplitFMNIST(
            dataset_root=DATASETS,
            n_experiences=10,
            first_batch_with_half_classes=False,
            shuffle=False,
        )

class MyExperiment(Experiment):

    def __init__(self, dropout_rate) -> None:
        self.dropout_rate = dropout_rate

        super().__init__()

    def make_network(self) -> nn.Module:
        return MyNetwork(**self.log_hparam(dropout_rate=self.dropout_rate))
    
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

    def add_plugins(self) -> Sequence[StrategyPlugin]:
        return [
            ReplayPlugin(**self.log_hparam(mem_size=100), storage_policy=
                ClassBalancedBuffer(
                    max_size=100, 
                    adaptive_size=False, 
                    total_num_classes=self.scenario.n_classes))
        ]
    # def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
    #     # Too much forgetting happens because we give it lots of epochs todo it
    #     # in order to make dropouts effect more obvious we add this
    #     if strategy.training_exp_counter > 0:
    #         self.strategy.train_epochs = 1


    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, **self.log_hparam(lr=0.01))

    def configure_regime(self) -> dict:
        return dict(train_mb_size = 100, train_epochs = 5, eval_mb_size = 1000)

    def make_scenario(self):
        return dataset


search_space = np.arange(0.0, 0.99, 0.01)
print(search_space)

random.shuffle(search_space)
# for droprate in search_space:
MyExperiment(0.5).train()