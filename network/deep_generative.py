from dataclasses import dataclass
import typing
from torch import Tensor, nn
import torch
from torch.nn import functional as F
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

    def forward(self, x: Tensor) -> AutoEncoder.ForwardOutput:
        z = self.encoder(x)      # Latent codes
        x_hat = self.decoder(z)  # Reconstruction
        y_hat = self.head(z)     # Classification head
        return AutoEncoder.ForwardOutput(y_hat, x, x_hat, z)

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

    @dataclass
    class ForwardOutput(AutoEncoder.ForwardOutput):
        mu: Tensor      # means
        log_var: Tensor # log(variance)


    def forward(self, input: Tensor, **kwargs) -> ForwardOutput:
        mu, log_var = self.encode(input)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decode(z)

        y_hat = self.class_head(z)
        return DVAE.ForwardOutput(
            y_hat, input, x_hat, z, mu, log_var)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def sample_z(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

class DAE_Loss():
    """
    Discriminative Auto-Encoder loss function

    2x joint objectives:
     * Classifier accuracy
     * Minimise reconstruction error
    """

    def __init__(self,
        recon_weight: float,
        classifier_weight: float) -> None:
        self.recon_weight = recon_weight
        self.classifier_weight = classifier_weight 

    def _recon_loss(self, x_hat, x):
        loss = F.mse_loss(x_hat, x, reduction="none")
        # Mean sum of pixel differences
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return self.recon_weight * loss

    def _classifier_loss(self, y, y_hat):
        return self.classifier_weight * F.cross_entropy(y_hat, y)
    
    def loss(self, out: DAE.ForwardOutput, y: Tensor):
        return self._recon_loss(out.x_hat, out.x) + \
               self._classifier_loss(y, out.y_hat)


class DVAE_Loss(DAE_Loss):
    """
    Discriminative Variational Auto-Encoder loss function. 
    
    3x joint objectives:
     * Classifier accuracy
     * Minimise reconstruction error
     * Make latent space follow gaussian distributions
    """

    def __init__(self,
        recon_weight: float,
        classifier_weight: float,
        kld_weight: float
        ) -> None:
        super().__init__(recon_weight, classifier_weight)
        self.kld_weight = kld_weight

    def _kl_loss(self, mu, log_var):
        # KL loss if we assume a normal distribution!
        return self.kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    def loss(self, out: DVAE.ForwardOutput, y: Tensor) -> Tensor:
        kl = self._kl_loss(out.mu, out.log_var)
        recon = self._recon_loss(out.x_hat, out.x)
        classifier = self._classifier_loss(y, out.y_hat)

        # print(f"kl: {kl}, recon: {recon}, classifier: {classifier}")
        return  kl + recon + classifier


