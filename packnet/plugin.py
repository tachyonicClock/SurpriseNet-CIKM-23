import typing as t
from avalanche.core import SupervisedPlugin
from torch import nn
import numpy as np
from click import secho


from experiment.experiment import BaseExperiment


def equal_capacity_prune_schedule(n_experiences: int) -> t.List[float]:
    """Returns a pruning schedule that results in equal capacity for each experience.
    For example if there are 4 experAiences, the schedule will be [3/4, 2/3, 1/2].
    Note that only the remaining capacity is pruned, so each task will have the same
    capacity as the first task.

    :param n_experiences: The number of tasks to generate a schedule for
    :return: A list of pruning proportions
    """
    schedule = []
    for i in range(n_experiences):
        schedule.append((n_experiences-1-i)/(n_experiences-i))
    return schedule

class PackNetPlugin(SupervisedPlugin):
    """Plugin that calls PackNet functionality during training"""

    def __init__(
            self,
            prune_amount: t.Union[float, t.List[float]],
            post_prune_epochs: int):
        """Create a plugin to add PackNet functionality to an experiment

        :param network: The network to prune
        :param experiment: The experiment to add the plugin to
        :param prune_amount: The proportion of the network to prune. 
            If a list, the proportion to prune at each experience
        :param post_prune_epochs: The number of epochs to train after pruning
        """
        self.prune_amount = prune_amount
        self.post_prune_epochs = post_prune_epochs
        self.enabled = True
        self.capacity: float = 1.0
        """How much of the network is still trainable"""

    def after_training_exp(self, strategy, **kwargs):
        """Perform pruning"""
        if not self.enabled:
            return
        network = strategy.model

        if isinstance(self.prune_amount, float):
            prune_proportion = self.prune_amount
        elif isinstance(self.prune_amount, list):
            prune_proportion = self.prune_amount[strategy.experience.current_experience]
        else:
            raise ValueError("prune_schedule must be a float or a list")

        self.capacity *= prune_proportion
        
        secho(f"Pruning {prune_proportion*100:0.1f}% reclaiming {self.capacity*100:0.1f}% capacity", fg="green")
        network.prune(prune_proportion)

        # Retrain post prune
        for _ in range(self.post_prune_epochs):
            strategy._before_training_epoch(**kwargs)
            strategy.training_epoch()
            strategy._after_training_epoch(**kwargs)

        secho("Freezing Task-Specific Subset", fg="green")
        network.push_pruned()
