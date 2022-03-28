
from dataclasses import dataclass
from torch import Tensor
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.models import avalanche_forward


@dataclass
class ForwardOutput():
    y_hat: Tensor
    """The predicted class label"""
    x: Tensor = None
    """The original input"""
    x_hat: Tensor = None
    """Reconstruction"""
    z_code: Tensor = None
    """The models internal and compressed representation"""
    pred_exp_id: Tensor = None
    """The predicted exp id for a forward pass"""
    exp_id: Tensor = None
    """The actual exp id for a forward pass"""
    mu: Tensor = None
    """The output of the mean layer in a VAE"""
    log_var: Tensor = None
    """The output of the variance layer in a VAE"""

class Strategy(SupervisedTemplate):
    """
    Strategy inherits from avalanche's `SupervisedTemplate`. It allows
    for greater modification than just plugins. An avalanche strategy is
    responsible for activating callbacks and for the flow of data between
    subsystems.
    """

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        self.last_forward_output: ForwardOutput = \
            avalanche_forward(self.model, self.mb_x, self.mb_task_id)
        return self.last_forward_output.y_hat


    @property
    def step(self) -> int:
        return self.clock.train_iterations
