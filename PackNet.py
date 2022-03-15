import random
import torch
import torch.nn as nn
from torch import Tensor
from experiment.experiment import Experiment, BaseHyperParameters
from network.trait import PackNetModule
from network.module.packnet_linear import PackNetLinear

from conf import *

from dataclasses import dataclass
from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST
import avalanche as av


class SimplePackNet(PackNetModule):

    def __init__(self):
        super().__init__()

        def layer(in_features, out_features) -> nn.Module:
            return nn.Sequential(
                PackNetLinear(in_features, out_features),
                nn.ReLU(),
            )

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            layer(28*28, 256),
            nn.Dropout(0.5),
            layer(256, 128),
            nn.Dropout(0.5),
            layer(128, 100),
            nn.Dropout(0.5),
            layer(100, 10),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


@dataclass
class HyperParams(BaseHyperParameters):
    post_prune_epochs: int
    prune_proportion: float
    prune: bool


class MyExperiment(Experiment):

    hp: HyperParams
    network: SimplePackNet

    classifier_weight: float = 50.0

    def __init__(self, hp: HyperParams) -> None:
        super().__init__(hp)

        self.after_eval_forward = self.after_forward

    def make_network(self) -> nn.Module:
        return SimplePackNet()

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(parameters, self.hp.lr)
        return optimizer

    def make_criterion(self):
        self.cross_entropy = nn.CrossEntropyLoss()
        return self.cross_entropy

    capacity: float = 1.0
    """How much of the network is still trainable"""
    def after_training_exp(self, strategy):
        if not self.hp.prune:
            return
        
        self.capacity *= self.hp.prune_proportion
        print("Performing Prune")
        print(f"     Pruning     {self.hp.prune_proportion}")
        print(f"     New Capcity {self.capacity}")
        self.network.prune(self.hp.prune_proportion)

        for _ in range(self.hp.post_prune_epochs):
            self.strategy.training_epoch()

        # exit(0)
        print("Push Pruned")
        self.network.push_pruned()


    def before_eval_exp(self, strategy, *args, **kwargs):
        experience: av.benchmarks.Experience = self.strategy.experience
        task_id = experience.task_label

        if self.hp.prune:
            print(f"task_id={task_id}, experience={self.clock.train_exp_counter}")
            if task_id >= self.clock.train_exp_counter:
                self.network.use_top_subset()
            else:
                self.network.use_task_subset(task_id)
    
    def after_eval(self, strategy, *args, **kwargs):
        self.network.use_top_subset()

    def make_scenario(self):
        scenario = SplitFMNIST(
            n_experiences=5,
            fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            return_task_id=True,
            dataset_root=DATASETS)
        return scenario

# Set seeds for reproducibility
random.seed(0)
torch.manual_seed(42)

experiment = MyExperiment(
    HyperParams(
            lr=0.005,
            train_mb_size=64,
            train_epochs=1,
            eval_mb_size=128,
            eval_every=-1,
            post_prune_epochs=5,
            prune_proportion=0.5,
            prune=True,
            device="cuda",
    )
)

result = experiment.train()

print("done!")