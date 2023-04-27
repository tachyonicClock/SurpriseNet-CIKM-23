
from collections import defaultdict
from dataclasses import dataclass

import torch
from avalanche.training import Cumulative
from avalanche.training.templates import SupervisedTemplate
from torch import Tensor, nn


@dataclass
class ForwardOutput():
    """
    Forward output is a big ugly class used to keep track of all the details 
    that various objects might wish to use.
    """

    y_hat: Tensor = None
    """The predicted class label"""
    y: Tensor = None
    """The actual class label"""
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

    We use this class to add ForwardOutput to the avalanche strategy. Allowing
    us to pass around multiple outputs of the model in a single object. This
    is useful for auto-encoders.
    """

    batch_transform: nn.Module = nn.Identity()
    """
    Transform the input before passing it to the model. Used for generating
    embeddings on the fly.
    """

    def forward(self):
        """Compute the model's output given the current mini-batch."""

        # Perform transformation without recording gradient
        with torch.no_grad():
            self.mbatch[0] = self.batch_transform(self.mb_x)

        self.last_forward_output: ForwardOutput = self.model.multi_forward(
            self.mbatch[0])
        self.last_forward_output.x = self.mb_x
        self.last_forward_output.y = self.mb_y

        return self.last_forward_output.y_hat

    @property
    def step(self) -> int:
        return self.clock.train_iterations

    def reset_optimizer(self):
        """Reset the optimizer"""
        self.optimizer.state = defaultdict(dict)


class CumulativeTraining(Cumulative, Strategy):
    pass
