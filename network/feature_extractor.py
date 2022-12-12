from typing import List
from avalanche.benchmarks import NCScenario, nc_benchmark
from torch import nn, Tensor
import torch
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50FeatureExtractor(nn.Module):
    """A feature extractor using ResNet"""

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.eval()

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

def wrap_with_resize(size: int, func):
    """Interpolate a tensor to a new size"""
    def _wrap(x: torch.Tensor):
        x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        return func(x)
    return _wrap

class SmallResNet18(nn.Module):
    """Patch torchvision.models.resnet18 to work with smaller images, such as 
    TinyImageNet (64x64)
    """

    def __init__(self, num_classes: int, pretrained: bool = False):
        super(SmallResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained)

        # Patch early layers that overly downsample the image
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet.maxpool = nn.Identity()

        # Patch the final layer to output the correct number of classes
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        # print(x.shape)
        # x = self.resnet.layer4(x)
        x = torch.flatten(x, 1)
        return x

def small_r18_extractor(save: str) -> nn.Module:
    """Create a feature extractor from a pre-trained SmallResNet18"""
    model = SmallResNet18(num_classes=200, pretrained=False)
    model.load_state_dict(torch.load(save))
    model.resnet.fc = nn.Identity()
    return model
    
def r18_extractor() -> nn.Module:
    """Create a feature extractor from a pre-trained ResNet18"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    # model.forward = wrap_with_resize(224, model.forward)
    return model
    