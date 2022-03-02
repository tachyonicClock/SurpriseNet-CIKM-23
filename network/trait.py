from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from torch import Tensor, nn
from avalanche.core import SupervisedPlugin


class HasFeatureMap():
    def forward_to_featuremap(self, input: Tensor) -> Tensor:
        raise NotImplemented

    def get_backbone(self) -> nn.Module:
        raise NotImplemented

class TaskAware():
    """
    ML algorthims that should be told about the task they are in during
    training
    """

    def on_task_change(self, new_task_id: int):
        """
        on_task_change is called before new task data starts being fed to 
        the network

        Args:
            new_task_id (int): An id that uniquely represents the task.
        """
        pass

class Generative(ABC):
    '''Generative algorithms with classification capability'''

    @dataclass
    class ForwardOutput():
        y_hat: Tensor
        """The models classification predictions"""
        x_hat: Tensor
        """The generative models reconstruction"""

        z_code: Tensor
        """The generative models internal representation"""

    @abstractclassmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    @abstractclassmethod
    def decode(self, z: Tensor) -> Tensor:
        pass

    @abstractclassmethod
    def sample_z(self, n: int=0) -> Tensor:
        """Sample the latent dimension and generate `n` encodings"""
        pass

    @abstractclassmethod
    def classify(self, x: Tensor) -> Tensor:
        pass

class SpecialLoss(ABC):

    @abstractclassmethod
    def loss_function(self, *args, **kwargs):
        pass

class TraitPlugin(SupervisedPlugin):
    """
    The trait plugin implements the trait behaviors using avalanche
    """

    def before_training_exp(self, strategy, **kwargs):
        if isinstance(strategy.model, TaskAware):
            print("ON TASK CHANGE", strategy.clock.train_exp_counter)
            strategy.model.on_task_change(strategy.clock.train_exp_counter)
