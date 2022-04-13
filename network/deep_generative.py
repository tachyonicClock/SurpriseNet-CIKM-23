from dataclasses import dataclass
from turtle import forward
import typing
from torch import Tensor, nn
import torch
from torch.nn import functional as F

from experiment.strategy import ForwardOutput
from .trait import Classifier, Decoder, Samplable, Encoder, AutoEncoder


class DAE(AutoEncoder, Classifier, nn.Module):
    """
    Discriminative Auto-Encoder
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        classifier: Classifier) -> None:
        super().__init__()

        assert encoder.bottleneck_width == decoder.bottleneck_width, \
            f"The encoder({encoder.bottleneck_width}) and decoder must have the same bottleneck width ({decoder.bottleneck_width})"

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder.encode(x)
    
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder.decode(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    @property
    def bottleneck_width(self) -> int:
        return self.encoder.bottleneck_width

    def forward(self, x: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        out.z_code = self.encoder.encode(x)
        out.x_hat = self.decoder.decode(out.z_code)
        out.y_hat = self.classifier.classify(out.z_code)
        out.x = x
        return out


# class DAE(AutoEncoder, Classifier, nn.Module):
#     """
#     Discriminative Auto-Encoder
#     """

#     def __init__(self,
#                  latent_dim: int,
#                  encoder: nn.Module,
#                  decoder: nn.Module,
#                  head: nn.Module) -> None:
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.encoder: nn.Module = encoder
#         self.decoder: nn.Module = decoder
#         self.head: nn.Module = head

#     def encode(self, x: Tensor):
#         return self.encoder(x)

#     def decode(self, z: Tensor):
#         return self.decoder(z)

#     def classify(self, x: Tensor) -> Tensor:
#         return self.forward(x).y_hat

#     def forward(self, x: Tensor) -> ForwardOutput:
#         z = self.encoder(x)      # Latent codes
#         x_hat = self.decoder(z)  # Reconstruction
#         y_hat = self.head(z)     # Classification head
#         return ForwardOutput(y_hat=y_hat, x=x, x_hat=x_hat, z_code=z)

# class DVAE(Classifier, Samplable, AutoEncoder, nn.Module):
#     """
#     Discriminative Variational Auto-Encoder

#     Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
#     """

#     def __init__(self,
#         latent_dim: int,
#         encoder: nn.Module,
#         decoder: nn.Module,
#         class_head: nn.Module,
#         mu_head: nn.Module,
#         var_head: nn.Module
#         ) -> None:
#         """_summary_

#         :param latent_dim: The size of the latent dimension
#         :param encoder: Pattern encoded and fed through mu and var heads
#         :param decoder: Takes latent code and creates reconstruction  
#         :param class_head: Outputs classes
#         :param mu_head: Outputs means of latent distribution
#         :param var_head: Outputs variances means of latent distribution
#         """
#         super().__init__()

#         self.latent_dim = latent_dim
#         self.mu_head    = mu_head
#         self.var_head   = var_head
#         self.class_head = class_head
#         self.encoder    = encoder
#         self.decoder    = decoder


#     def encode(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
#         x = self.encoder(x)
#         return self.mu_head(x), self.var_head(x)

#     def decode(self, z: Tensor) -> Tensor:
#         x = self.decoder(z)
#         return x

#     def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
#         # Do the "Reparameterization Trick" aka sample from the distribution
#         std = torch.exp(0.5 * log_var)  # 0.5 square roots it
#         eps = torch.randn_like(std)
#         z = mu + eps*std
#         return z


#     def forward(self, input: Tensor, **kwargs) -> ForwardOutput:
#         mu, log_var = self.encode(input)
#         z = self.reparameterise(mu, log_var)
#         x_hat = self.decode(z)

#         y_hat = self.class_head(z)
#         return ForwardOutput(
#             y_hat=y_hat,
#             x_hat=x_hat,
#             z_code=z,
#             mu=mu,
#             log_std=log_var)

#     def classify(self, x: Tensor) -> Tensor:
#         return self.forward(x).y_hat

#     def sample_z(self, n:int=1) -> Tensor:
#         return torch.randn(n, self.latent_dim)

#     @property
#     def device(self):
#         return next(self.parameters()).device
