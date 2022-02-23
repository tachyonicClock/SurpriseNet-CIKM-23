from torch import Tensor, nn
from avalanche.core import BasePlugin, SupervisedPlugin


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


class IsGenerative():
    '''ML algorithms with a generative component can inherit from this class'''

    def encode(self, x: Tensor):
        pass

    def decode(self, z: Tensor):
        pass


class TraitPlugin(BasePlugin):
    """
    The trait plugin implements the trait behaviors using avalanche
    """

    def before_eval_exp(self, strategy: SupervisedPlugin, **kwargs):
        if isinstance(strategy.model, TaskAware):
            strategy.model.on_task_change(strategy.clock.train_exp_counter)
