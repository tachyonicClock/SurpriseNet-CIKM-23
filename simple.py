from dataclasses import dataclass
import torch
import torchvision as tv
from torch import Tensor, nn

from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST
from avalanche.benchmarks.classic.ccifar10 import SplitCIFAR10

from experiment.experiment import Experiment, BaseHyperParameters
from experiment.loss import  ClassifierLoss, MultipleObjectiveLoss, ReconstructionError, VAELoss
from experiment.plugins import PackNetPlugin
from network.components.classifier import PN_ClassifyHead
from network.components.decoder import PN_CNN_Decoder
from network.components.encoder import PN_CNN_Encoder
from network.components.sampler import PN_VAE_Sampler
from network.deep_generative import DAE, DVAE, PN_DAE_InferTask, PN_DVAE_InferTask
import config

@dataclass
class HyperParams(BaseHyperParameters):
    prune_proportion: float
    post_prune_epoch: int
    network_type: str
    input_channels: int
    base_channel_size: int
    vae_beta: float
    class_weight: float

    vae_bottleneck: int
    ae_bottleneck: int

class SimpleExperiment(Experiment):
    hp: HyperParams
    network: DVAE

    def __init__(self, hp: HyperParams) -> None:
        super().__init__(hp)


    def make_network(self) -> nn.Module:
        hp = self.hp
        if self.hp.network_type == "VAE":
            network = PN_DVAE_InferTask(
                PN_CNN_Encoder(hp.input_channels, hp.base_channel_size, hp.ae_bottleneck),
                PN_VAE_Sampler(hp.ae_bottleneck, hp.vae_bottleneck),
                PN_CNN_Decoder(hp.input_channels, hp.base_channel_size, hp.vae_bottleneck),
                PN_ClassifyHead(hp.vae_bottleneck, self.n_classes)
            )
        elif self.hp.network_type == "AE":
            network = PN_DAE_InferTask(
                PN_CNN_Encoder(hp.input_channels, hp.base_channel_size, hp.bottleneck_width),
                PN_CNN_Decoder(hp.input_channels, hp.base_channel_size, hp.bottleneck_width),
                PN_ClassifyHead(hp.bottleneck_width, self.n_classes)
            )
            pass
        else:
            assert False, f"Network type `{self.hp.network_type}` unknown"
        return network
    

    def add_plugins(self):
        return [PackNetPlugin(self, self.hp.prune_proportion, self.hp.post_prune_epoch)]

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(parameters, self.hp.lr)
        return optimizer

    def make_objective(self) -> MultipleObjectiveLoss:
        loss = MultipleObjectiveLoss().add(ReconstructionError()) \
                                      .add(ClassifierLoss(self.hp.class_weight))

        if self.hp.network_type == "VAE":
            loss.add(VAELoss(self.hp.vae_bottleneck, 3*32*32*self.hp.train_mb_size, self.hp.vae_beta))

        return loss
              

    def make_scenario(self):
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize((0.2860,), (0.3530,)), 
            tv.transforms.RandomHorizontalFlip(), 
            tv.transforms.Resize((32, 32))
        ])

        # scenario = SplitFMNIST(
        #     n_experiences=5,
        #     fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     dataset_root=config.DATASETS,
        #     return_task_id=False,
        #     eval_transform=transform,
        #     train_transform=transform
        # )

        scenario = SplitCIFAR10(
            n_experiences=5,
            fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dataset_root=config.DATASETS,
            return_task_id=False,
            # first_exp_with_half_classes=True
        )

        return scenario


SimpleExperiment(
    HyperParams(
        lr=0.0005,
        vae_beta=0.001,
        class_weight=0,

        network_type="VAE",

        prune_proportion=0.5,
        
        train_mb_size=128,
        eval_mb_size=128,

        train_epochs=500,
        post_prune_epoch=250,

        ae_bottleneck=500,
        vae_bottleneck=128,
        base_channel_size=32,
        input_channels=3,

        eval_every=-1,
        device="cuda"
    )
).train()