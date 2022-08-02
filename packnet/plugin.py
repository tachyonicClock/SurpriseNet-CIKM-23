from avalanche.core import SupervisedPlugin

from experiment.experiment import BaseExperiment

class PackNetPlugin(SupervisedPlugin):
    """Plugin that implements the steps to implement PackNet"""
    capacity: float = 1.0
    """How much of the network is still trainable"""

    def __init__(self, network, experiment: BaseExperiment, prune_proportion, post_prune_epochs):
        self.network = network
        self.experiment = experiment
        self.prune_proportion = prune_proportion
        self.post_prune_epochs = post_prune_epochs

    def after_training_exp(self, strategy, **kwargs):
        """Perform pruning"""
        self.capacity *= self.prune_proportion

        print(f"Pruning Network, {self.capacity*100}% remaining")
        self.network.prune(self.prune_proportion)

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
    

