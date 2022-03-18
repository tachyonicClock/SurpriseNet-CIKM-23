from collections import OrderedDict
import copy
import typing
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from .trait import AutoEncoder, Classifier
from .coders import *


def MLP_AE_Head(latent_dims, output_size):
    return nn.Sequential(
        nn.Linear(latent_dims, output_size),
        nn.ReLU()
    )


class D_AE(AutoEncoder, Classifier, nn.Module):
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
        self.cross_entropy = nn.CrossEntropyLoss()

    # Overload trait.IsGenerative functions
    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def sample_z(self, n) -> Tensor:
        dist  = torch.distributions.Uniform(0.0, 1.0)
        return dist.sample((n, self.latent_dim))

    def forward(self, x: Tensor) -> AutoEncoder.ForwardOutput:
        """ Forward through the neural network
        Returns:
            typing.Tuple[Tensor, Tensor]: Reconstruction and predicted class
                labels
        """
        z = self.encoder(x)      # Latent codes
        x_hat = self.decoder(z)  # Reconstruction
        y_hat = self.head(z)     # Classification head
        return AutoEncoder.ForwardOutput(y_hat, x_hat, z)


# class AEGroup(AutoEncoder, TaskAware, SpecialLoss, nn.Module):
#     """A group of autoencoders"""
#     current_task: int = 0
#     encoders: typing.Sequence[Generative] = {}

#     def __init__(self, 
#         make_auto_encoder: typing.Callable[[], Generative],
#         n: int,
#         copy_network: bool = False):
#         super().__init__()
#         self.n_encoders = n
#         self.encoders = [make_auto_encoder() for _ in range(n)]
#         self._encoders = nn.ModuleList(self.encoders)
#         self.make_auto_encoder = make_auto_encoder
#         self.copy_network = copy_network

#     def on_task_change(self, new_task_id: int):

#         if self.copy_network:
#             print("COPY NETWORK")
#             old = self.encoders[self.current_task]
#             new = self.encoders[new_task_id]
#             new.load_state_dict(copy.deepcopy(old.state_dict()))

#         assert new_task_id < self.n_encoders, \
#             f"Got new task id {new_task_id} while only having {self.n_encoders}"
#         self.current_task = new_task_id

#     def forward(self, x: Tensor) -> Generative.ForwardOutput:
#         # If training we know the task
#         if self.training:
#             return self.get_encoder().forward(x)
#         return self._eval_forward(x)


#     def _training_forward(self, x: Tensor) -> Generative.ForwardOutput:
#         return self.get_encoder().forward(x)

#     def best_reduce(self, metric: Tensor, values: Tensor) -> Tensor:
#         """
#         Using an at least 2D metric array we want to return a tensor concatinating
#         all the best results
#         """
#         best = metric.T.argmin(dim=(1))
#         tensors = torch.stack([values[b][i] for i, b in enumerate(best)])
#         return tensors

#     @torch.no_grad()
#     def _eval_forward(self, x: Tensor) -> Generative.ForwardOutput:
#         # print(x)

#         # Loop over all encoders
#         metric, x_hats, y_hats, z_codes = [], [], [], []
#         for task, encoder in enumerate(self.encoders):
#             out = encoder.forward(x)

#             loss = F.mse_loss(x, out.x_hat, reduction="none") \
#                     .sum(dim=[1, 2, 3])

#             metric.append(loss)
#             y_hats.append(out.y_hat)
#             x_hats.append(out.x_hat)
#             z_codes.append(out.z_code)

#         metric = torch.stack(metric)
#         # print(x.shape)
#         x_hat  = self.best_reduce(metric, torch.stack(x_hats))
#         y_hat  = self.best_reduce(metric, torch.stack(y_hats))
#         z_code = self.best_reduce(metric, torch.stack(z_codes))


#         return self.ForwardOutput(y_hat, x_hat, z_code)

#     def get_encoder(self) -> typing.TypeVar("T", SpecialLoss, Generative):
#         return self.encoders[self.current_task]

#     def loss_function(self, *args, **kwargs):
#         return self.get_encoder().loss_function(*args, **kwargs)

#     def sample_z(self, n: int=0) -> Tensor:
#         """Sample the latent dimension and generate `n` encodings"""
#         return self.get_encoder().sample_z(n)

#     def encode(self, x: Tensor) -> Tensor:
#         return self.forward(x).z_code

#     def decode(self, z: Tensor) -> Tensor:
#         return self.get_encoder().decode(z)

#     def classify(self, x: Tensor) -> Tensor:
#         return self.forward(x).y_hat

#     # def parameters(self, recurse: bool = True) -> typing.Iterator[torch.Parameter]:
#     #     for x in self.
