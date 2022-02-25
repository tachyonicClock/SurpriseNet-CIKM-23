"""
Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch import Tensor, long
from network.coders import MLP_Encoder, MLP_Decoder

from network.trait import Generative


def kl_divergence():
    torch.kl_div()


class VAE(Generative, nn.Module):

    kld_weight = 0.005

    def __init__(self) -> None:
        super().__init__()

        self.latent_adjacent_dims = 64 # Dimensions of the layer before the latent dim
        self.latent_dims = 64 # Dimensions of the latent space


        # Fully connected layer which represents means
        self.fc_mu = nn.Linear(self.latent_adjacent_dims, self.latent_dims)
        # Fully connected layer which representing variances
        self.fc_var = nn.Linear(self.latent_adjacent_dims, self.latent_dims)


        self.decoder_adapter = nn.Linear(self.latent_dims, self.latent_adjacent_dims)

        self.encoder = MLP_Encoder(self.latent_adjacent_dims)
        self.decoder = MLP_Decoder(self.latent_adjacent_dims)


    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, z: Tensor) -> Tensor:
        # print(z.shape)
        x = self.decoder_adapter.forward(z)
        x = self.decoder(z)
        return x


    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # Do the "Reparameterization Trick" aka sample from the distribution
        std = torch.exp(0.5 * log_var)  # 0.5 square roots it
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    @dataclass
    class ForwardOutput(Generative.ForwardOutput):
        x: Tensor       # input
        mu: Tensor      # means
        log_var: Tensor # log(variance)


    def forward(self, input: Tensor, **kwargs) -> ForwardOutput:
        mu, log_var = self.encode(input)
        z = self.reparameterise(mu, log_var)
        recons = self.decode(z)

        y_hat = torch.zeros(len(input), dtype=torch.int).cuda()
        return VAE.ForwardOutput(y_hat, recons, input, mu, log_var)

    def classify(self, x: Tensor) -> Tensor:
        return torch.zeros(len(x), dtype=torch.int).cuda()

    def sample_z(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dims)

    @property
    def device(self):
        return next(self.parameters()).device


    def loss_function(self, recons: Tensor, input: Tensor, mu: Tensor, log_var: Tensor, y=None):

        recons_loss = F.mse_loss(recons, input)

        # Kullback–Leibler divergence how similar is the sample distribution to a normal distribution
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        return recons_loss + self.kld_weight * kld_loss

