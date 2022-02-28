"""
Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch import Tensor, long, tensor
from network.coders import MLP_Encoder, MLP_Decoder

from network.trait import Generative


def kl_divergence():
    torch.kl_div()


class VAE(Generative, nn.Module):

    kld_weight = 0.0001
    classifier_weight = 1.0
    cross_entropy = nn.CrossEntropyLoss()

    def __init__(self) -> None:
        super().__init__()

        self.latent_adjacent_dims = 64 # Dimensions of the layer before the latent dim
        self.latent_dims = 64 # Dimensions of the latent space


        # Fully connected layer which represents means
        self.fc_mu = nn.Linear(self.latent_adjacent_dims, self.latent_dims)
        # Fully connected layer which representing variances
        self.fc_var = nn.Linear(self.latent_adjacent_dims, self.latent_dims)

        self.fc_classifier = nn.Linear(self.latent_dims, 10)

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

        y_hat = self.fc_classifier(z)
        return VAE.ForwardOutput(y_hat, recons, input, mu, log_var)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def sample_z(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dims)

    @property
    def device(self):
        return next(self.parameters()).device


    def loss_function(self, out: ForwardOutput, y: Tensor):

        recons_loss = F.mse_loss(out.x_hat, out.x)

        # Kullback–Leibler divergence how similar is the sample distribution to a normal distribution
        kld_loss = torch.mean(-0.5 * torch.sum(1 + out.log_var - out.mu ** 2 - out.log_var.exp(), dim = 1), dim = 0)

        

        return recons_loss + self.kld_weight * kld_loss +  self.cross_entropy(out.y_hat, y)* self.classifier_weight

