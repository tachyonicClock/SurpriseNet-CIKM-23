import typing
from torch import nn, Tensor
from torch.nn import functional as F
from network.trait import Sampler, PackNetComposite
import network.module.packnet as pn


class VAE_Sampler(Sampler, nn.Module):

    def __init__(self, input_width: int, latent_dims: int) -> None:
        super().__init__()
        self.mu = nn.Linear(input_width, latent_dims)
        self.log_var = nn.Linear(input_width, latent_dims)
        self.latent_dims = latent_dims

    @property
    def bottleneck_width(self) -> int:
        return self.latent_dims

    def encode(self, input: Tensor) -> typing.Tuple[Tensor, Tensor]:
        input = F.relu(input)
        return self.mu(input), self.log_var(input)
    

class PN_VAE_Sampler(VAE_Sampler, PackNetComposite):

    def __init__(self, input_width: int, latent_dims: int) -> None:
        super().__init__(input_width, latent_dims)
        self.mu = pn.wrap(self.mu)
        self.log_var = pn.wrap(self.log_var)