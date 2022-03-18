"""
Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch import Tensor, long, tensor
from functional import vae_kl_loss
from network.module.packnet_linear import PackNetDenseDecoder, PackNetDenseEncoder

from network.trait import AutoEncoder, Classifier, Samplable


def kl_divergence():
    torch.kl_div()


class VAE(Classifier, Samplable, AutoEncoder, nn.Module):

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

        self.encoder = PackNetDenseEncoder((1, 28, 28), 64, [512, 256, 128])
        self.decoder = PackNetDenseDecoder((1, 28, 28), 64, [128, 256, 512])


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
    class ForwardOutput(AutoEncoder.ForwardOutput):
        x: Tensor       # input
        mu: Tensor      # means
        log_var: Tensor # log(variance)


    def forward(self, input: Tensor, **kwargs) -> ForwardOutput:
        mu, log_var = self.encode(input)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decode(z)

        y_hat = self.fc_classifier(z)
        return VAE.ForwardOutput(y_hat, x_hat, z, input, mu, log_var)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def sample_z(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dims)

    @property
    def device(self):
        return next(self.parameters()).device

class VAE_Loss():

    def __init__(self,
        kld_weight: float,
        recon_weight: float,
        classifier_weight: float) -> None:
        self.kld_weight = kld_weight
        self.recon_weight = recon_weight
        self.classifier_weight = classifier_weight 

    def _recon_loss(self, x, x_hat):
        # per-pixel mse
        recon_loss = F.mse_loss(x, x_hat, reduction="none")
        # Use the batch mean sum of the per-pixel mse 
        recon_loss = recon_loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return self.recon_weight * recon_loss

    def _kl_loss(self, mu, log_var):
        return self.kld_weight * vae_kl_loss(mu, log_var)

    def _classifier_loss(self, y, y_hat):
        return self.classifier_weight * F.cross_entropy(y_hat, y)

    def loss(self, out: VAE.ForwardOutput, y: Tensor) -> Tensor:
        return self._recon_loss(out.x, out.x_hat) + \
               self._kl_loss(out.mu, out.log_var) + \
               self._classifier_loss(y, out.y_hat)
