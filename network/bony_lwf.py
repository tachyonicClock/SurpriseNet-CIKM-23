"""
LWF is traditionally  used on the output layer. However it is likely better to
be used on a feature map instead. Called bony because it has a backbone
"""

from torch import Tensor
import torch.nn as nn

from network.nn_traits import HasConditionedDropout, HasFeatureMap
from network.module.dropout import ConditionedDropout



class BonyLWF(nn.Module, HasFeatureMap, HasConditionedDropout):

    def __init__(self, n_groups, p_active, p_inactive) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
             
        dropout_layer = ConditionedDropout(100, n_groups, p_active, p_inactive)
        self.register_conditioned_dropout_layer(dropout_layer)

        self.head = nn.Sequential(
            nn.Linear(64, 100),
            dropout_layer,
            nn.Linear(100, 10),
            nn.ReLU(),
        )
    
    def forward_to_featuremap(self, input: Tensor):
        return self.backbone.forward(input)

    def freeze_backbone(self):
        """Freeze the backbone"""
        print("Freeze backbone")
        for theta in self.backbone.parameters():
            theta.requires_grad = False

    def forward(self, input: Tensor):
        return self.head.forward(self.backbone.forward(input))
