from typing import List
from avalanche.benchmarks import NCScenario, nc_benchmark
from torch import nn, Tensor
import torch
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms as T


class ResNet50FeatureExtractor(nn.Module):
    """A feature extractor using ResNet"""

    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.eval()
        self.normalize = T.Normalize(mean, std)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x: Tensor):
        x = self.normalize(x)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x        
