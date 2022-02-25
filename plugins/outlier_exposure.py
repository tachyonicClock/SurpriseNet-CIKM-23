from typing import Optional, TYPE_CHECKING

from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    AvalancheConcatDataset,
)


class OutlierExposure(ExemplarsBuffer):
    """ABC for rehearsal buffers to store exemplars.

    `self.buffer` is an AvalancheDataset of samples collected from the previous
    experiences. The buffer can be updated by calling `self.update(strategy)`.
    """

    def __init__(self, max_size: int):
        """Init.

        :param max_size: max number of input samples in the replay memory.
        """
        self.__init__(max_size)

    # @abstractmethod
    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update `self.buffer` using the `strategy` state.

        :param strategy:
        :param kwargs:
        :return:
        """
        pass

    # @abstractmethod
    def resize(self, strategy: "SupervisedTemplate", new_size: int):
        """Update the maximum size of the buffer.

        :param strategy:
        :param new_size:
        :return:
        """
        pass