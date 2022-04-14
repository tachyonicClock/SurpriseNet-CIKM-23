from abc import ABC, abstractmethod
from dataclasses import dataclass
from turtle import forward
from typing import Callable
import typing
from torch import Tensor, nn
import torch
from experiment.strategy import ForwardOutput, Strategy


class PackNetParent(PackNet, nn.Module):
    def _pn_apply(self, func: Callable[['PackNet'], None]):
        @torch.no_grad()
        def __pn_apply(module):
            # Apply function to all child packnets but not other parents.
            # If we were to apply to other parents we would duplicate
            # applications to their children
            if isinstance(module, PackNet) and not isinstance(module, PackNetParent):
                func(module)
        self.apply(__pn_apply)

    def prune(self, to_prune_proportion: float) -> None:
        self._pn_apply(lambda x : x.prune(to_prune_proportion))

    def push_pruned(self) -> None:
        self._pn_apply(lambda x : x.push_pruned())

    def use_task_subset(self, task_id):
        self._pn_apply(lambda x : x.use_task_subset(task_id))

    def use_top_subset(self):
        self._pn_apply(lambda x : x.use_top_subset())
    

class Classifier(ABC):

    @abstractmethod
    def classify(self, x: Tensor) -> Tensor:
        pass


class Samplable(ABC):

    @abstractmethod
    def sample_z(self, n: int = 1) -> Tensor:
        pass


class Encoder(ABC):
    """"""

    @property
    @abstractmethod
    def bottleneck_width(self) -> Tensor:
        """The width of the bottleneck"""

    def encode(self, x: Tensor) -> Tensor:
        """Encode x to z

        :param x: An input
        :return: A tensor of the same shape as the bottleneck
        """

class Decoder(nn.Module, ABC):
    """A decoder for an AE"""

    @property
    @abstractmethod
    def bottleneck_width(self) -> Tensor:
        """The width of the output"""

    def decode(self, z: Tensor) -> Tensor:
        """Decode z to z

        :param z: A latent code
        :return: A tensor 
        """

class ProbabilisticEncoder(Encoder):
    """
    Something that encodes something using a normal distribution as the 
    latent space
    """

    @abstractmethod
    def reparameterise(self, mu: Tensor, log_std: Tensor) -> Tensor:
        # Do the "Reparameterization Trick" aka sample from the distribution
        std = torch.exp(0.5 * log_std)  # 0.5 square roots it
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z


    def encode_parameters(self, input: Tensor) -> typing.Tuple[Tensor, Tensor]:
        """Encode an input and get parameters representing its position in latent
        space

        :param input: Input to encoder
        :return: A tuple of mu and log_std
        """

class DVAE(nn.Module, Samplable):

    def __init__(self,
        encoder: Encoder,
        probabilistic_encoder: ProbabilisticEncoder,
        decoder: Decoder,
        classifier: Classifier) -> None:
        super()

        assert encoder.bottleneck_width == probabilistic_encoder.input_width, \
            f"The bottleneck of the encoder ({encoder.bottleneck_width}) must " \
            f"be the same width as the input to the probabilistic encoder " \
            f"({probabilistic_encoder.input_width})"

        assert probabilistic_encoder.output_width == decoder.bottleneck_width, \
            f"The output of the probabilistic encoder decoder must be the same width as the decoder bottleneck"

        assert encoder.bottleneck_width == classifier.input_width, \
            f"The output of the encoder must be the same width as the classifier input"
        
        self.encoder = encoder
        self.probabilistic_encoder = probabilistic_encoder
        self.decoder = decoder
        self.classifier = classifier

    def forward(self, x: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        z = self.encoder.encode(x)
        out.mu, out.log_var = self.probabilistic_encoder.encode_parameters(z)
        out.z_code = self.probabilistic_encoder.reparameterise(out.mu, out.log_var)
        out.y_hat = self.classifier(z)
        out.x_hat = self.decoder(z)
        return out

    def sample_x(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dim)

def get_all_trait_types():
    return [
        PackNet,
        AutoEncoder,
        Classifier,
        Samplable
    ]