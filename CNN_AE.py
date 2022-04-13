import typing
from dataclasses import dataclass

import avalanche as av
import torch
import torch.nn.functional as F
from avalanche.benchmarks.classic.ccifar100 import SplitCIFAR100
from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10
from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST
from torch import Tensor, nn

from config import *
from experiment.experiment import BaseHyperParameters, Experiment
from experiment.plugins import PackNetPlugin
from experiment.strategy import ForwardOutput
from functional import best_reduce
from network.coders import (PackNetDenseHead, PN_CNN_Decoder, PN_CNN_Encoder)
from network.deep_generative import DAE, Loss
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

    def forward(self, x: Tensor) -> ForwardOutput:
        if self.training:
            return super().forward(x)
        return self._eval_forward(x)

    def push_pruned(self):
        super().push_pruned()
        self.subnet_count += 1

    @torch.no_grad()
    def _eval_forward(self, x: Tensor) -> ForwardOutput:

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
        losses: Tensor = torch.stack(losses)
        x_hat = best_reduce(losses, torch.stack(x_hats))
        y_hat = best_reduce(losses, torch.stack(y_hats))
        z_code = best_reduce(losses, torch.stack(z_codes))
        pred_exp_id = losses.T.argmin(dim=1) 
        self.use_top_subset()
        return ForwardOutput(
            y_hat=y_hat, x=x, x_hat=x_hat,
            z_code=z_code, pred_exp_id=pred_exp_id)


@dataclass
class HyperParams(BaseHyperParameters):
    prune: bool
    prune_proportion: float
    post_prune_epochs: int
    latent_dims: int
    base_channel_size: int
    classifier_weight: float
    sparsifying_weight: float
    input_channels: int



class MyExperiment(Experiment):

    hp: HyperParams
    network: DAE

    def __init__(self, hp: HyperParams) -> None:
        super().__init__(hp)

    def make_network(self) -> nn.Module:
        latent_dims = self.hp.latent_dims

        # encoder = PackNetDenseEncoder((1, 28, 28), 64, [512, 256, 128])
        # decoder = PackNetDenseDecoder((1, 28, 28), 64, [128, 256, 512])

        channels = self.hp.input_channels
        encoder = PN_CNN_Encoder(channels, self.hp.base_channel_size, latent_dims)
        decoder = PN_CNN_Decoder(channels, self.hp.base_channel_size, latent_dims)

        head = PackNetDenseHead(latent_dims, self.scenario.n_classes)
        return PackNetClassifyingAutoEncoder(latent_dims, encoder, decoder, head)

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(parameters, self.hp.lr)
        return optimizer

    def add_plugins(self):
        return [PackNetPlugin(self.network, self.hp.prune_proportion, self.hp.post_prune_epochs)]

    def make_mulitpart_loss(self) -> Loss:
        return Loss(
            classifier_weight=self.hp.classifier_weight, 
            recon_weight=1.0, 
            sparsifying_weight=self.hp.sparsifying_weight)

    def make_criterion(self):
        def _loss_function(output: Tensor, target: Tensor) -> Tensor:
            self.loss.update(self.last_mb_output, target)
            return self.loss.weighted_sum()
        return _loss_function

    def make_scenario(self):
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)), transforms.RandomHorizontalFlip(), transforms.Resize((32, 32))]
        # )
        # scenario = SplitFMNIST(
        #     n_experiences=5,
        #     fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     dataset_root=DATASETS,
        #     return_task_id=False,
        #     eval_transform=transform,
        #     train_transform=transform
        #     )

        # scenario = SplitCIFAR10(
        #     n_experiences=5,
        #     fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     dataset_root=DATASETS,
        #     return_task_id=False,
        #     )

        scenario = SplitCIFAR100(
            n_experiences=10,
            fixed_class_order=list(range(100)),
            dataset_root=DATASETS
        )

        return scenario

# FMNIST
# experiment = MyExperiment(
#     HyperParams(
#         lr=0.0005,
#         train_mb_size=64,
#         eval_mb_size=64,
#         eval_every=-1,
#         prune=True,
#         prune_proportion=0.50,

#         # Epochs
#         train_epochs=5,
#         post_prune_epochs=1,

#         # Loss weights
#         classifier_weight=500,
#         sparsifying_weight=0.1,


#         # Network architecture
#         base_channel_size=32,
#         latent_dims=64,
#         device="cuda",
#         input_channels=1,
#     )
# )

experiment = MyExperiment(
    HyperParams(
        lr=0.0002,
        train_mb_size=128,
        eval_mb_size=128,
        eval_every=-1,
        prune=True,
        prune_proportion=0.8,

        # Epochs
        train_epochs=1000,
        post_prune_epochs=500,

        # Loss weights
        classifier_weight=1000,
        sparsifying_weight=0.0,

        # Network architecture
        base_channel_size=64,
        latent_dims=512,
        input_channels=3,
        device="cuda"
    )
)

# experiment = MyExperiment(
#     HyperParams(
#         lr=0.0005,
#         train_mb_size=64,
#         train_epochs=500,
#         eval_mb_size=64,
#         eval_every=-1,
#         prune=True,
#         prune_proportion=0.5,
#         post_prune_epochs=100,
#         classifier_weight=1000,
#         sparsifying_weight=1,

#         # Network architecture
#         base_channel_size=32,
#         latent_dims=128,
#         device="cuda"
#     )
# )


# for k, v in experiment.network.named_parameters():
#     print(k)

experiment.train()
print("DONE!")
