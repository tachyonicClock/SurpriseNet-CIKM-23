from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Callable
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

    def _pn_apply(self, func: Callable[['PackNetModule'], None]):
        """Apply only to child PackNetModule"""
        @torch.no_grad()
        def __pn_apply(module):
            if isinstance(module, PackNetModule) and module != self:
                func(module)
        self.apply(__pn_apply)


    def prune(self, to_prune_proportion: float) -> None:
        """Prune a proportion of the prunable parameters using the absolute value
        of the weights as a heuristic for importance (Han et al., 2017)

        :param to_prune_proportion: A proportion of the prunable parameters to prune
        """
        self._pn_apply(lambda x : x.prune(to_prune_proportion))

    def push_pruned(self) -> None:
        """
        Moves pruned parameters to the top of the stack. Note that biases are 
        frozen as a side-effect.
        """
        self._pn_apply(lambda x : x.push_pruned())

    def use_task_subset(self, task_id):
        self._pn_apply(lambda x : x.use_task_subset(task_id))

    def use_top_subset(self):
        self._pn_apply(lambda x : x.use_top_subset())




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
