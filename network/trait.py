from abc import ABC, abstractmethod
import typing as t
from torch import Tensor, TensorType, nn
import torch
from avalanche.models.generator import Generator

from experiment.strategy import ForwardOutput

class Classifier(ABC, nn.Module):
    """Something that can classify"""

    @abstractmethod
    def classify(self, embedding: Tensor) -> Tensor:
        """x -> y_hat

        :param x: An batch of observations to classify
        :return: A class of predicted probabilities
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

class Samplable(Generator, ABC):
    """Something that can generate instances"""

    @abstractmethod
    def sample(self, n: int = 1) -> Tensor:
        """Sample from a generative model

        :param n: Number of samples to generate, defaults to 1
        :return: A generated sample
        """

    def generate(self, batch_size=None, condition=None):
        return self.sample(batch_size)
    
    def get_features(self):
        raise NotImplemented("get features not implemented")

class ConditionedSample(Samplable):
    """Something that can generate instances"""

    @abstractmethod
    def conditioned_sample(self, n: int = 1, given_task: int = 0) -> Tensor:
        pass

    def generate(self, batch_size=None, condition=None):
        return self.conditioned_sample(batch_size, condition)

class Sampler(Samplable, nn.Module):
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

    def forward(self, x: Tensor) -> t.Tuple[Tensor, Tensor]:
        return NotImplemented


class AutoEncoder(Encoder, Decoder, Classifier, nn.Module):

    def __init__(self,
        encoder: Encoder,
        decoder: Decoder,
        classifier: Classifier
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def classify(self, embedding: Tensor) -> Tensor:
        return self.classifier(embedding)

    def encode(self, observations: Tensor) -> Tensor:
        return self.encoder.encode(observations)

    def decode(self, embedding: Tensor) -> Tensor:
        return self.decoder.decode(embedding)

    def forward(self, observations: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        out.z_code = self.encode(observations)
        out.y_hat = self.classify(out.z_code)
        out.x_hat  = self.decode(out.z_code)
        return out

class VariationalAutoEncoder(AutoEncoder, Samplable):

    def __init__(self,
        encoder: Encoder,
        bottleneck: Sampler,
        decoder: Decoder,
        classifier: Classifier) -> None:
        super().__init__(encoder, decoder, classifier)
        self.dummy_param = nn.Parameter(torch.empty(0)) # Used to determine device
        self.bottleneck = bottleneck

    def encode(self, x: Tensor) -> Tensor:
        mu, std = self.bottleneck.encode(self.encoder.encode(x))
        return self.bottleneck.reparameterise(mu, std)
    
    def sample(self, n: int = 1) -> Tensor:
        return self.decode(self.bottleneck.sample(n).to(self.dummy_param.device))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder.decode(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    def forward(self, x: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        z = self.encoder.encode(x)
        out.mu, out.log_var = self.bottleneck.forward(z)
        out.z_code = self.bottleneck.reparameterise(out.mu, out.log_var)
        out.y_hat = self.classifier(out.z_code)
        out.x_hat = self.decoder(out.z_code)
        out.x = x
        return out

    def sample_x(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dim)


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
    VariationalAutoEncoder,
    InferTask
]