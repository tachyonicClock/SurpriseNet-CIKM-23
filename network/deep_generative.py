from dataclasses import dataclass
import typing
from torch import Tensor, nn
import torch
from torch.nn import functional as F

from experiment.strategy import ForwardOutput
from functional import recon_loss
from .trait import AutoEncoder, Classifier,Samplable


class DAE(AutoEncoder, Classifier, nn.Module):
    """
    Discriminative Auto-Encoder
    """

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

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def forward(self, x: Tensor) -> ForwardOutput:
        z = self.encoder(x)      # Latent codes
        x_hat = self.decoder(z)  # Reconstruction
        y_hat = self.head(z)     # Classification head
        return ForwardOutput(y_hat=y_hat, x=x, x_hat=x_hat, z_code=z)

class DVAE(Classifier, Samplable, AutoEncoder, nn.Module):
    """
    Discriminative Variational Auto-Encoder

    Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """

    def __init__(self,
        latent_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        class_head: nn.Module,
        mu_head: nn.Module,
        var_head: nn.Module
        ) -> None:
        """_summary_

        :param latent_dim: The size of the latent dimension
        :param encoder: Pattern encoded and fed through mu and var heads
        :param decoder: Takes latent code and creates reconstruction  
        :param class_head: Outputs classes
        :param mu_head: Outputs means of latent distribution
        :param var_head: Outputs variances means of latent distribution
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.mu_head    = mu_head
        self.var_head   = var_head
        self.class_head = class_head
        self.encoder    = encoder
        self.decoder    = decoder


    def encode(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        return self.mu_head(x), self.var_head(x)

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        return x

    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # Do the "Reparameterization Trick" aka sample from the distribution
        std = torch.exp(0.5 * log_var)  # 0.5 square roots it
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z



    def forward(self, input: Tensor, **kwargs) -> ForwardOutput:
        mu, log_var = self.encode(input)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decode(z)

        y_hat = self.class_head(z)
        return ForwardOutput(
            y_hat=y_hat,
            x_hat=x_hat,
            z_code=z,
            mu=mu,
            log_var=log_var)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def sample_z(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device




class LossPart():
    loss: Tensor = 0.0
    weighting: float = 1.0

    @property
    def weighted_loss(self):
        return self.weighting * self.loss

    @property
    def is_used(self):
        return self.weighting != 0.0

    def __init__(self, weighting: float) -> None:
        self.weighting = weighting

class Loss():
    """
    My loss class for calculating non-trivial multi-part loss functions
    """

    def __init__(self,
        classifier_weight: float = 0.0,
        recon_weight: float = 0.0,
        kld_weight: float = 0.0) -> None:

        self.classifier = LossPart(classifier_weight)
        self.recon = LossPart(recon_weight)
        self.kld = LossPart(kld_weight)

    def _kl_loss(self, mu, log_var):
        # KL loss if we assume a normal distribution!
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    def update(self, out: ForwardOutput, y: Tensor):
        if self.recon.is_used:
            self.recon.loss = recon_loss(out.x_hat, out.x)
        if self.classifier.is_used:
            self.classifier.loss = F.cross_entropy(out.y_hat, y)
        if self.kld.is_used:
            self.kld.loss = self._kl_loss(out.mu, out.log_var)

    def weighted_sum(self):
        return self.classifier.weighted_loss + self.recon.weighted_loss + self.kld.weighted_loss

