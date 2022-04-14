from abc import ABC, abstractmethod
import typing
from torch import Tensor, TensorType
import torch

class Classifier(ABC):
    """Something that can classify"""

    @abstractmethod
    def classify(self, x: Tensor) -> Tensor:
        """x -> x_hat

        :param x: An batch of observations to classify
        :return: A class count wide tensor of predicted probabilities
        """

class ClassifyExperience(ABC):
    """
    Does the classifier try to identify the experience of an observation
    """

    @abstractmethod
    def classify_experience(self, x: Tensor) -> Tensor:
        pass

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
    def conditioned_sample(self, n: int = 1, given_class: int = 0) -> Tensor:
        pass


class Encoder(ABC):
    """Something that can encode something"""

    @property
    @abstractmethod
    def bottleneck_width(self) -> int:
        """The width of the bottleneck"""

    def encode(self, x: Tensor) -> Tensor:
        """x -> z"""

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
    def encode(self, input: Tensor) -> typing.Tuple[Tensor, Tensor]:
        """Encode an input and get parameters representing its position in latent
        space

        :param input: Input to encoder
        :return: A tuple of mu and log_var
        """

class Decoder(ABC):
    """Something that can decoded something that was encoded"""

    @property
    @abstractmethod
    def bottleneck_width(self) -> int:
        """The width of the bottleneck"""

    def decode(self, z: Tensor) -> Tensor:
        """z -> x"""

class AutoEncoder(Encoder, Decoder):
    pass

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

class PackNetComposite(PackNet, torch.nn.Module):
    def _pn_apply(self, func: typing.Callable[['PackNet'], None]):
        @torch.no_grad()
        def __pn_apply(module):
            # Apply function to all child packnets but not other parents.
            # If we were to apply to other parents we would duplicate
            # applications to their children
            if isinstance(module, PackNet) and not isinstance(module, PackNetComposite):
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
    


NETWORK_TRAITS = [
    Classifier,
    Samplable,
    Encoder,
    Decoder,
    PackNet,
    AutoEncoder,
    ConditionedSample,
]