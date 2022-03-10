from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from torch import Tensor, nn
import torch
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

class PackNetModule(nn.Module):

    layer_count: int = 0

    def prune(self, to_prune_proportion: float) -> None:
        """Prune a proportion of the prunable parameters using the absolute value
        of the weights as a heuristic for importance (Han et al., 2017)

        :param to_prune_proportion: A proportion of the prunable parameters to prune
        """

        @torch.no_grad()
        def _prune(module: nn.Module):
            if isinstance(module, PackNetModule) and module != self:
                module.prune(to_prune_proportion)
        self.apply(_prune)

    def push_pruned(self) -> None:
        """
        Moves pruned parameters to the top of the stack. Note that biases are 
        frozen as a side-effect.
        """
        @torch.no_grad()
        def _push_pruned(module: nn.Module):
            if isinstance(module, PackNetModule) and module != self:
                module.push_pruned()
        self.apply(_push_pruned)
        self.layer_count += 1

    def task_id_to_z_index(self, task_id) -> int:
        return self.layer_count - task_id

    def task_forward(self, input: Tensor, z_index: int) -> Tensor:
        """Forward using a specified subnetwork

        :param z_index: Specify a subnetwork
        :param input: Activations that propagate forward
        :return: The output layer activations
        """
        return self.forward(input, z_index)
 

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
