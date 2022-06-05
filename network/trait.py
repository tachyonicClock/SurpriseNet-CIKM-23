from abc import ABC, abstractmethod
import typing as t
from torch import Tensor, TensorType, nn
import torch

from experiment.strategy import ForwardOutput

class Classifier(ABC, nn.Module):
    """Something that can classify"""

    @abstractmethod
    def classify(self, x: Tensor) -> Tensor:
        """x -> x_hat

        :param x: An batch of observations to classify
        :return: A class count wide tensor of predicted probabilities
        """

class Encoder(ABC, nn.Module):

    @abstractmethod
    def encode(self, observations: Tensor) -> Tensor:
        """Encodes observations into an embedding

        :param observation: A tensor to encode
        :return: An embedding
        """
        return NotImplemented

class Decoder(ABC, nn.Module):

    @abstractmethod
    def decode(self, embedding: Tensor) -> Tensor:
        """Decodes an embedding into a reconstruction

        :param embedding: An encoded observation
        :return: A reconstruction
        """
        return NotImplemented


class InferTask():
    """
    Does the classifier try to identify the experience of an observation
    """

class Samplable(ABC):
    """Something that can generate instances"""

    @abstractmethod
    def sample(self, n: int = 1) -> Tensor:
        """Sample from a generative model

        :param n: Number of samples to generate, defaults to 1
        :return: A generated sample
        """

class ConditionedSample(ABC):
    """Something that can generate instances"""

    @abstractmethod
    def conditioned_sample(self, n: int = 1, given_task: int = 0) -> Tensor:
        pass



class Sampler(Encoder, Samplable):
    """
    Something that encodes something using a normal distribution as the 
    latent space
    """

    def reparameterise(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # Do the "Reparameterization Trick" aka sample from the distribution
        std = torch.exp(0.5 * log_var)  # 0.5 square roots it
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def sample(self, n: int = 1) -> Tensor:
        return torch.randn((n, self.bottleneck_width))

    @abstractmethod
    def encode(self, input: Tensor) -> t.Tuple[Tensor, Tensor]:
        """Encode an input and get parameters representing its position in latent
        space

        :param input: Input to encoder
        :return: A tuple of mu and log_var
        """


class AutoEncoder(Encoder, Decoder, nn.Module):

    def __init__(self,
        encoder: Encoder,
        decoder: Decoder
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.encode = encoder.encode
        self.decode = decoder.decode

    def encode(self, observations: Tensor) -> Tensor:
        return self.encoder.encode(observations)

    def decode(self, embedding: Tensor) -> Tensor:
        return self.encoder.decode(embedding)

    def forward(self, observations: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        out.z_codes = self.encoder.encode(observations)
        out.x_hat  = self.decoder.decode(out.z_codes)
        return out

class PackNet(ABC):

    @abstractmethod
    def prune(self, to_prune_proportion: float) -> None:
        """Prune a proportion of the prunable parameters (parameters on the 
        top of the stack) using the absolute value of the weights as a 
        heuristic for importance (Han et al., 2017)

        :param to_prune_proportion: A proportion of the prunable parameters to prune
        """

    @abstractmethod
    def push_pruned(self) -> None:
        """
        Commits the layer by incrementing counters and moving pruned parameters
        to the top of the stack. Biases are frozen as a side-effect.
        """

    @abstractmethod
    def use_task_subset(self, task_id):
        pass

    @abstractmethod
    def use_top_subset(self):
        pass




NETWORK_TRAITS = [
    Classifier,
    Samplable,
    Encoder,
    Decoder,
    PackNet,
    AutoEncoder,
    ConditionedSample,
    InferTask
]