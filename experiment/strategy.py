
from dataclasses import dataclass
from torch import Tensor
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.models import avalanche_forward



class Strategy(SupervisedTemplate):
    """
    Strategy inherits from avalanche's `SupervisedTemplate`. It allows
    for greater modification than just plugins. An avalanche strategy is
    responsible for activating callbacks and for the flow of data between
    subsystems.
    """

    @dataclass
    class ForwardOutput():
        y_hat: Tensor


    def forward(self):
        """Compute the model's output given the current mini-batch."""
        self.last_forward_output: Strategy.ForwardOutput = \
            avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        return self.last_forward_output.y_hat