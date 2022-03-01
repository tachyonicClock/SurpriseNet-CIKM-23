from collections import OrderedDict
import typing
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from .trait import Generative, SpecialLoss, TaskAware
import torchvision.transforms as transforms
from .coders import *


def MLP_AE_Head(latent_dims, output_size):
    return nn.Sequential(
        nn.Linear(latent_dims, output_size),
        nn.ReLU()
    )


class D_AE(Generative, SpecialLoss):
    # Discriminative auto-encoder

    def __init__(self,
                 latent_dim: int,
                 classifier_weight: float,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 head: nn.Module) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.classifier_weight = classifier_weight
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.head: nn.Module = head
        self.cross_entropy = nn.CrossEntropyLoss()

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
        return D_AE.ForwardOutput(y_hat, x_hat)

    def _reconstruction_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def _classifier_loss(self, y: Tensor, y_hat: Tensor) -> Tensor:
        return self.cross_entropy(y_hat, y)
    
    def loss_function(self, x, y, x_hat, y_hat):
        return self._reconstruction_loss(x, x_hat) + \
               self._classifier_loss(y, y_hat) * self.classifier_weight


class AEGroup(Generative, TaskAware, SpecialLoss):
    """A group of autoencoders"""
    current_task: int = 0
    encoders: typing.Dict[int, Generative] = {}

    def __init__(self, make_auto_encoder: typing.Callable[[], Generative]):
        self.encoders[0] = make_auto_encoder()
        self.make_auto_encoder = make_auto_encoder

    def on_task_change(self, new_task_id: int):
        self.current_task = new_task_id

        if new_task_id not in self.encoders.keys():
            self.encoders[new_task_id] = self.make_auto_encoder()

    def forward(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        # If training we know the task
        if self.training:
            return self.get_encoder().forward(x)


        raise NotImplemented


    def _training_forward(self, x: Tensor) -> Generative.ForwardOutput:
        return self.get_encoder().forward(x)

    def best_reduce(metric: Tensor, values: Tensor) -> Tensor:
        """
        Using an at least 2D metric array we want to return a tensor concatinating
        all the best results
        """
        best = metric.argmax(dim=(1))
        tensors = Tensor([x[i] for i, x in zip(best, values)])
        return tensors

    @torch.no_grad()
    def _eval_forward(self, x: Tensor) -> Generative.ForwardOutput:

        # Loop over all encoders
        metric, x_hats, y_hats = [], [], []
        for task, encoder in self.encoders.items():
            out = encoder.forward(x)

            loss = F.mse_loss(x, out.x_hat, reduction="none") \
                    .sum(dim=[1, 2, 3])

            metric.append(loss)
            y_hats.append(out.y_hat)
            x_hats.append(out.x_hat)

        metric = Tensor(metric)
        x_hat = self.best_reduce(metric, Tensor(x_hats))
        y_hat = self.best_reduce(metric, Tensor(y_hats))

    def get_encoder(self):
        return self.encoders[self.current_task]

    def loss(self, forward_output: Generative.ForwardOutput, target):
        raise NotImplemented
