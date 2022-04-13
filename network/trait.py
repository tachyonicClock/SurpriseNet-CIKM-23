from abc import ABC, abstractmethod
from torch import Tensor, TensorType

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
    def classify(self, x: Tensor) -> Tensor:
        pass

class Samplable(ABC):
    """Something that can generate instances"""

    @abstractmethod
    def sample(self, n: int = 1) -> Tensor:
        """Sample from a generative model

        :param n: Number of samples to generate, defaults to 1
        :return: A generated sample
        """

class Encoder(ABC):
    """Something that can encode something"""

    @property
    @abstractmethod
    def bottleneck_width(self) -> int:
        """The width of the bottleneck"""

    def encode(self, x: Tensor) -> Tensor:
        """x -> z"""

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

NETWORK_TRAITS = [
    Classifier,
    Samplable,
    Encoder,
    Decoder,
    PackNet,
    AutoEncoder
]