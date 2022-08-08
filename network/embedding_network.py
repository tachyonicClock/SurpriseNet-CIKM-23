from avalanche.benchmarks import NCScenario, nc_benchmark
from torch import nn, Tensor
import torch
from torchvision import models
from torch.utils.data import Dataset

class ResNet18FeatureExtractor(nn.Module):
    """A feature extractor using ResNet"""

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.eval()

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
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
