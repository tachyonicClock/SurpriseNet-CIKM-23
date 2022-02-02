from typing import Sequence
from torch import Tensor
from torch import nn
from network.module.dropout import ConditionedDropout

class HasFeatureMap():
    def forward_to_featuremap(self, input: Tensor) -> Tensor:
        raise NotImplemented

    def get_backbone(self) -> nn.Module:
        raise NotImplemented

class HasConditionedDropout():

    n_groups: int = None
    _callbacks = []

    def register_conditioned_dropout_layer(self, layer: ConditionedDropout) -> ConditionedDropout:
        self.n_groups = self.n_groups or layer.n_groups # If none set
        assert self.n_groups == layer.n_groups, "Cannot register layers with different number of groups"
        self._callbacks.append(layer.set_active_group)

    def set_active_group(self, group: int):
        [f(group) for f in self._callbacks]



    