from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from torch import Tensor, nn
import torch
from experiment.strategy import Strategy


class PackNetModule(nn.Module):

    def _pn_apply(self, func: Callable[['PackNetModule'], None]):
        """Apply only to child PackNetModule"""
        @torch.no_grad()
        def __pn_apply(module):
            if isinstance(module, PackNetModule) and module != self:
                func(module)
        self.apply(__pn_apply)

    def prune(self, to_prune_proportion: float) -> None:
        """Prune a proportion of the prunable parameters (parameters on the 
        top of the stack) using the absolute value of the weights as a 
        heuristic for importance (Han et al., 2017)

        :param to_prune_proportion: A proportion of the prunable parameters to prune
        """
        self._pn_apply(lambda x : x.prune(to_prune_proportion))

    def push_pruned(self) -> None:
        """
        Commits the layer by incrementing counters and moving pruned parameters
        to the top of the stack. Biases are frozen as a side-effect.
        """
        self._pn_apply(lambda x : x.push_pruned())

    def use_task_subset(self, task_id):
        self._pn_apply(lambda x : x.use_task_subset(task_id))

    def use_top_subset(self):
        self._pn_apply(lambda x : x.use_top_subset())

class AutoEncoder(ABC):
    '''Generative algorithms with classification capability'''

    @dataclass
    class ForwardOutput(Strategy.ForwardOutput):
        x_hat: Tensor
        """The generative models reconstruction"""
        z_code: Tensor
        """The generative models internal representation"""

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        pass


class Classifier(ABC):

    @abstractmethod
    def classify(self, x: Tensor) -> Tensor:
        pass


class Samplable(ABC):

    @abstractmethod
    def sample_z(self, n: int = 1) -> Tensor:
        pass



def get_all_trait_types():
    return [
        PackNetModule,
        AutoEncoder,
        Classifier,
        Samplable
    ]