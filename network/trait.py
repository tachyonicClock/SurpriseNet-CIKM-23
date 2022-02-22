import typing
from matplotlib.pyplot import cla
from torch import Tensor, nn
from network.module.dropout import ConditionedDropout
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy


class HasFeatureMap():
    def forward_to_featuremap(self, input: Tensor) -> Tensor:
        raise NotImplemented

    def get_backbone(self) -> nn.Module:
        raise NotImplemented

class TaskAware():
    """ML algorthims that should be told about the task they are in during training"""

    def on_task_change(self, new_task_id: int):
        """on_task_change is called before new task data starts being fed to the network

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


class TraitPlugin(StrategyPlugin):
    """
    The trait plugin implements the trait behaviors using avalanche
    """
    def before_eval_exp(self, strategy: BaseStrategy, **kwargs):
        if isinstance(strategy.model, TaskAware):
            strategy.model.on_task_change(strategy.clock.train_exp_counter)





    