from dataclasses import dataclass
from typing import Sequence

import avalanche as av
import numpy as np
import torch
from avalanche.benchmarks.classic.cmnist import RotatedMNIST
from avalanche.evaluation.metrics.accuracy import accuracy_metrics
from avalanche.evaluation.metrics.confusion_matrix import \
    confusion_matrix_metrics
from avalanche.evaluation.metrics.forgetting_bwt import forgetting_metrics
from avalanche.evaluation.metrics.loss import LossPluginMetric, loss_metrics
from avalanche.training.plugins import ReplayPlugin, StrategyPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.plugins.synaptic_intelligence import \
    SynapticIntelligencePlugin
from torch import Tensor, device, nn
from torchvision.transforms import transforms

from conf import *
from experiment import BaseHyperParameters, Experiment
from metrics.featuremap import FeatureMap
from metrics.metrics import TrainExperienceLoss
from network.bony_lwf import BonyLWF
from plugins.BackboneLWF import BackboneLWF

# from avalanche.training.plugins.lwf import LwFPlugin




dataset = av.benchmarks.classic.SplitFMNIST(
            dataset_root=DATASETS,
            n_experiences=6,
            first_batch_with_half_classes=True,
            shuffle=False,
            fixed_class_order=[0,1,2,3,4,5,6,7,8,9],
        )

@dataclass
class HyperParams(BaseHyperParameters):
    lr: float
    train_mb_size: int
    train_epochs: int
    eval_mb_size: int

    lwf_alpha: float
    si_lambda: float
    lfl_lambda: float

class MyExperiment(Experiment):

    hp: HyperParams
    network: BonyLWF

    def __init__(self, hp: HyperParams) -> None:
        super().__init__(hp)

    def make_network(self) -> nn.Module:
        return BonyLWF(self.n_experiences, self.hp.p_active, self.hp.p_inactive)
    


    def add_plugins(self) -> Sequence[StrategyPlugin]:
        return [BackboneLWF(alpha=self.hp.alpha, temperature=self.hp.temperature)]

    def before_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        experience = self.strategy.clock.train_exp_counter
        print("activating group", experience)
        self.network.set_active_group(experience)

    # def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
    #     # self.schedule.step()
    #     self.log_scalar("lr", self.lr, self.strategy.clock.train_iterations)

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters), self.hp.lr)
        # self.schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8)
        return optimizer

    def make_scenario(self):
        return dataset

def do_experiment():
    hp = HyperParams(
        lr=np.random.uniform(0.0001, 1),
        train_mb_size=64,
        eval_mb_size=2**11,
        train_epochs=1,
        eval_every=-1,
        device="cuda",
        p_active=np.random.uniform(0.0, 1.0),
        p_inactive=np.random.uniform(0.0, 1.0),
        temperature=np.random.uniform(0.0, 10.0),
        alpha=np.random.uniform(0.0, 10.0)
    )

    MyExperiment(hp).train()

for _ in range(1000):
    do_experiment()
