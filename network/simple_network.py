import torch.nn as nn

from avalanche.models.dynamic_modules import MultiTaskModule, \
    MultiHeadClassifier
from avalanche.models.base_model import BaseModel


class SimpleDropoutMLP(nn.Module, BaseModel):

    def __init__(self, 
                num_classes=10, 
                input_size=28 * 28,
                hidden_size=512, 
                hidden_layers=1,
                dropout_module: nn.Module = nn.Dropout()):

        super().__init__()

        layers = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True),
                                 dropout_module)
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True),
                      dropout_module)))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x
