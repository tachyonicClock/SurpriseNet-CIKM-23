import typing
from dataclasses import dataclass

import avalanche as av
import torch
import torch.nn.functional as F
from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST
from torch import Tensor, nn

from conf import *
from experiment.experiment import BaseHyperParameters, Experiment
from experiment.plugins import PackNetPlugin
from functional import best_reduce
from network.coders import (PackNetDenseHead, PN_CNN_Decoder, PN_CNN_Encoder)
from network.deep_generative import DAE, DAE_Loss
from network.trait import PackNetParent

from torchvision.transforms import transforms


class PackNetClassifyingAutoEncoder(DAE, PackNetParent):
    encoder: PackNetParent
    decoder: PackNetParent
    head: PackNetParent

    subnet_count: int = 0

    def _pn_apply(self, func: typing.Callable[['PackNetParent'], None]):
        """Apply only to child PackNetModule"""
        for module in [self.encoder, self.decoder, self.head]:
            func(module)

    def forward(self, x: Tensor) -> DAE.ForwardOutput:
        if self.training:
            return super().forward(x)
        return self._eval_forward(x)

    def push_pruned(self):
        super().push_pruned()
        self.subnet_count += 1

    @torch.no_grad()
    def _eval_forward(self, x: Tensor) -> DAE.ForwardOutput:

        # Generate `ForwardOutput` for each `PackNet` subnetwork/stack layer
        losses: typing.Sequence[float] = []
        x_hats, y_hats, z_codes = [], [], []
        for i in range(self.subnet_count):
            self.use_task_subset(i)

            output = super().forward(x)
            # Calculate MSE to determine the how familiar the stack layer is with
            # the instance
            loss = F.mse_loss(
                output.x_hat, x, reduction="none").sum(dim=[1, 2, 3])

            x_hats.append(output.x_hat)
            y_hats.append(output.y_hat)
            z_codes.append(output.z_code)
            losses.append(loss)

        # Select the best subnetwork's output
        losses = torch.stack(losses)
        x_hat = best_reduce(losses, torch.stack(x_hats))
        y_hat = best_reduce(losses, torch.stack(y_hats))
        z_code = best_reduce(losses, torch.stack(z_codes))

        self.use_top_subset()

        return self.ForwardOutput(y_hat, x, x_hat, z_code)


@dataclass
class HyperParams(BaseHyperParameters):
    prune: bool
    prune_proportion: float
    post_prune_epochs: int
    classifier_weight: float = 1.0
    latent_dims: int = 64


class MyExperiment(Experiment):

    hp: HyperParams
    network: DAE

    def __init__(self, hp: HyperParams) -> None:
        super().__init__(hp)

    def make_network(self) -> nn.Module:
        latent_dims = self.hp.latent_dims

        # encoder = PackNetDenseEncoder((1, 28, 28), 64, [512, 256, 128])
        # decoder = PackNetDenseDecoder((1, 28, 28), 64, [128, 256, 512])

        channels = 1
        base_channel_size = 32
        encoder = PN_CNN_Encoder(channels, base_channel_size, latent_dims)
        decoder = PN_CNN_Decoder(channels, base_channel_size, latent_dims)

        head = PackNetDenseHead(latent_dims, 10)
        return PackNetClassifyingAutoEncoder(latent_dims, encoder, decoder, head)

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(parameters, self.hp.lr)
        return optimizer

    def add_plugins(self):
        return [PackNetPlugin(self.network, self.hp.prune_proportion, self.hp.post_prune_epochs)]

    def make_criterion(self):
        loss = DAE_Loss(1.0, self.hp.classifier_weight)

        def _loss_function(output: Tensor, target: Tensor) -> Tensor:
            return loss.loss(self.last_mb_output, target)
        return _loss_function

    def make_scenario(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)), transforms.RandomHorizontalFlip(), transforms.Resize((32, 32))]
        )
        scenario = SplitFMNIST(
            n_experiences=5,
            fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dataset_root=DATASETS,
            return_task_id=False,
            eval_transform=transform,
            train_transform=transform
            )
        print(scenario.n_classes)
        return scenario


experiment = MyExperiment(
    HyperParams(
        lr=0.0001,
        train_mb_size=64,
        train_epochs=10,
        eval_mb_size=32,
        eval_every=-1,
        prune=True,
        prune_proportion=0.5,
        post_prune_epochs=10,
        classifier_weight=10.0,
        device="cuda"
    )
)

experiment.train()
print("DONE!")
