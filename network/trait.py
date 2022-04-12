from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from torch import Tensor, nn
import torch
from experiment.strategy import Strategy

class PackNet(ABC):

    @abstractmethod
    def prune(self, to_prune_proportion: float) -> None:
        """Prune a proportion of the prunable parameters (parameters on the 
        top of the stack) using the absolute value of the weights as a 
        heuristic for importance (Han et al., 2017)

        :param to_prune_proportion: A proportion of the prunable parameters to prune
        """
        self._pn_apply(lambda x : x.prune(to_prune_proportion))

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


class AutoEncoder(ABC):
    '''Generative algorithms with classification capability'''
    
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


class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass

class LatentSampler(Samplable):
    pass

class DVAE():

    def __init__(self,
        encoder: Encoder,
        decorder: Decoder,
        latent_sampler: LatentSampler,
        classifier) -> None:
        pass

def get_all_trait_types():
    return [
        PackNet,
        AutoEncoder,
        Classifier,
        Samplable
    ]