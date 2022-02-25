import typing
from torch import Tensor, nn
import torch
from .trait import Generative, TaskAware
import torchvision.transforms as transforms
from .coders import *


def MLP_AE_Head(latent_dims, output_size):
    return nn.Sequential(
        nn.Linear(latent_dims, output_size),
        nn.ReLU()
    )


class D_AE(Generative):
    # Discriminative auto-encoder

    def __init__(self,
                 latent_dim: int,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 head: nn.Module) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.head: nn.Module = head

    # Overload trait.IsGenerative functions
    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def sample_z(self) -> Tensor:
        dist  = torch.distributions.Uniform(0.0, 1.0)
        return dist.sample((1, self.latent_dim))

    def forward(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        """ Forward through the neural network
        Returns:
            typing.Tuple[Tensor, Tensor]: Reconstruction and predicted class
                labels
        """
        z = self.encoder(x)      # Latent codes
        x_hat = self.decoder(z)  # Reconstruction
        y_hat = self.head(z)     # Classification head
        return Generative.ForwardOutput(y_hat, x_hat)


class AEGroup(Generative, TaskAware):
    """A group of autoencoders"""
    current_task: int = 0
    encoders: typing.Dict[int, nn.Module] = {}

    def __init__(self, make_auto_encoder: typing.Callable[[], nn.Module]):
        self.encoders[0] = make_auto_encoder()
        self.make_auto_encoder = make_auto_encoder

    def on_task_change(self, new_task_id: int):
        self.current_task = new_task_id

        if new_task_id not in self.encoders.keys():
            self.encoders[new_task_id] = self.make_auto_encoder()

    def forward(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        # If training we know the task
        if self.training:
            return self.get_encoder()(x)

        # TODO use task with lowest reconstruction loss
        raise NotImplemented("WIP ae_group:85")

    def get_encoder(self):
        return self.encoders[self.current_task]
