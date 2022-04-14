import torch
import torchvision as tv
from torch import Tensor, nn

from avalanche.benchmarks.classic.cfashion_mnist import SplitFMNIST

from experiment.experiment import Experiment, BaseHyperParameters
from experiment.loss import  MultipleObjectiveLoss, ReconstructionError, VAELoss
from network.components.classifier import PN_ClassifyHead
from network.components.decoder import PN_CNN_Decoder
from network.components.encoder import PN_CNN_Encoder
from network.components.sampler import PN_VAE_Sampler
from network.deep_generative import DAE, DVAE
import config

class HyperParams(BaseHyperParameters):
    bottleneck_width: int = 32
    input_channels: int = 1
    base_channel_size: int = 64

class SimpleExperiment(Experiment):
    hp: HyperParams
    network: DVAE

    def __init__(self, hp: HyperParams) -> None:
        super().__init__(hp)


    def make_network(self) -> nn.Module:
        hp = self.hp
        # network = DAE(
        #     CNN_Encoder(hp.input_channels, hp.base_channel_size, hp.bottleneck_width),
        #     CNN_Decoder(hp.input_channels, hp.base_channel_size, hp.bottleneck_width),
        #     ClassifyHead(hp.bottleneck_width, self.n_classes)
        # )

        network = DVAE(
            PN_CNN_Encoder(hp.input_channels, hp.base_channel_size, hp.bottleneck_width*2),
            PN_VAE_Sampler(hp.bottleneck_width*2, hp.bottleneck_width),
            PN_CNN_Decoder(hp.input_channels, hp.base_channel_size, hp.bottleneck_width),
            PN_ClassifyHead(hp.bottleneck_width, self.n_classes)
        )
        return network
        

    def make_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(parameters, self.hp.lr)
        return optimizer

    def make_objective(self) -> MultipleObjectiveLoss:
        return MultipleObjectiveLoss().add(ReconstructionError()) \
                                      .add(VAELoss(self.hp.bottleneck_width, 32*32*self.hp.train_mb_size, 1.0))
                                    #   .add(ClassifierLoss())

    def make_scenario(self):
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize((0.2860,), (0.3530,)), 
            tv.transforms.RandomHorizontalFlip(), 
            tv.transforms.Resize((32, 32))
        ])

        scenario = SplitFMNIST(
            n_experiences=5,
            fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dataset_root=config.DATASETS,
            return_task_id=False,
            eval_transform=transform,
            train_transform=transform
        )

        return scenario


SimpleExperiment(
    HyperParams(
        lr=0.002,
        train_mb_size=256,
        eval_mb_size=256,

        train_epochs=20,

        eval_every=-1,
        device="cuda"
    )
).train()