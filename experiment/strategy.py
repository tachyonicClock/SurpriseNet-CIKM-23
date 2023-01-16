
from dataclasses import dataclass
from typing import Protocol

from avalanche.models import avalanche_forward
from avalanche.training.templates import SupervisedTemplate
from torch import Tensor, nn
import avalanche as cl
import torch

@dataclass
class ForwardOutput():
    """
    Forward output is a big ugly class used to keep track of all the details 
    that various objects might wish to use.
    """

    y_hat: Tensor = None
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
    loss_by_layer: Tensor = None
    """
    When using task inference we calculate the loss for each layer in the PackNet.
    We store those here to create metrics. A tensor of shape (n_experiences, batch_size)
    containing the loss for the appropriate instance
    """

    # NOTE: This is a hack to trick the learning without forgetting plugin
    # to work with ForwardOutput since it expects the y_hat tensor instead
    def __getitem__(self, i):
         return self.y_hat[i]

class Network(Protocol):
    def forward(self, x: Tensor) -> ForwardOutput:
        pass

class Strategy(SupervisedTemplate):
    """
    Strategy inherits from avalanche's `SupervisedTemplate`. It allows
    for greater modification than just plugins. An avalanche strategy is
    responsible for activating callbacks and for the flow of data between
    subsystems.
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

        self.last_forward_output: ForwardOutput = self.model.multi_forward(self.mbatch[0])
        self.last_forward_output.x = self.mb_x
        return self.last_forward_output.y_hat


    @property
    def step(self) -> int:
        return self.clock.train_iterations

class CumulativeTraining(cl.training.Cumulative, Strategy):
    pass
