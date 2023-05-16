import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models


def r18_extractor() -> nn.Module:
    """Create a feature extractor from a pre-trained ResNet18"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()

    """ResNet18 feature extractor using `model.eval()`, `torch.no_grad()`, and 
    requires_grad=False` performs much better 5% -> 78%. Forcing it into eval
    mode in particular is important. My guess is the model is not put into eval
    mode before eval creating issues todo with the batch normalisation not
    dealing with class incremental evaluation well.
    """

    @torch.no_grad()
    def _forward(x: Tensor) -> Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return model._forward_impl(x).detach()

    model.forward = _forward
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
