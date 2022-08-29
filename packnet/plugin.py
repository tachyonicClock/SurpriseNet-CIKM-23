import typing as t
from avalanche.core import SupervisedPlugin
from torch import nn
import numpy as np


from experiment.experiment import BaseExperiment


def equal_capacity_prune_schedule(n_experiences: int) -> t.List[float]:
    target_proportion = 1.0 / (n_experiences+1)
    return list(np.arange(1.0-target_proportion, 0.0, -target_proportion))

class PackNetPlugin(SupervisedPlugin):
    """Plugin that implements the steps to implement PackNet"""
    capacity: float = 1.0
    """How much of the network is still trainable"""

    def __init__(
            self, 
            network: nn.Module,
            experiment: BaseExperiment,
            prune_amount: t.Union[float, t.List[float]],
            post_prune_epochs: int):
        """Create a plugin to add PackNet functionality to an experiment

        :param network: The network to prune
        :param experiment: The experiment to add the plugin to
        :param prune_amount: The proportion of the network to prune. 
            If a list, the proportion to prune at each experience
        :param post_prune_epochs: The number of epochs to train after pruning
        """
        self.network = network
        self.experiment = experiment
        self.prune_amount = prune_amount
        self.post_prune_epochs = post_prune_epochs
        print(f"Configured prune: {prune_amount}")

    def after_training_exp(self, strategy, **kwargs):
        """Perform pruning"""
        # If list
        if isinstance(self.prune_amount, float):
            prune_proportion = self.prune_amount
        elif isinstance(self.prune_amount, list):
            prune_proportion = self.prune_amount[strategy.experience.current_experience]
        else:
            raise ValueError("prune_schedule must be a float or a list")

        self.capacity *= prune_proportion
        
        print(f"Pruning {prune_proportion*100:0.2f} of network such that {self.capacity*100:0.2f}% remains unfrozen")
        self.network.prune(prune_proportion)

        # Reset optimizer
        self.experiment.optimizer = self.experiment.make_optimizer(self.network.parameters())
        strategy.optimizer = self.experiment.optimizer

        for _ in range(self.post_prune_epochs):
            strategy._before_training_epoch(**kwargs)
            strategy.training_epoch()
            strategy._after_training_epoch(**kwargs)

        print("Pushing")
        self.network.push_pruned()

        # Reset optimizer
        self.experiment.optimizer = self.experiment.make_optimizer(self.network.parameters())
        strategy.optimizer = self.experiment.optimizer


    # def before_training_exp(self, strategy, *args, **kwargs):
    #     """Reset for new experience"""
    #     self.network.use_top_subset()

    # def before_eval_exp(self, strategy: Strategy, *args, **kwargs):
    #     """Use task id to select the right part of each layer for eval"""
    #     task_id = strategy.experience.current_experience
    #     log.info(f"using task {task_id}")
    #     self.network.use_task_subset(task_id)
    
