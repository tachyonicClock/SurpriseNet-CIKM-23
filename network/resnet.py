"""
Based on https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
and on ResNet18

"""

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from network.trait import Decoder, Encoder


class BlockA(nn.Module):
    """
    Block A is a residual block with a shortcut connection. Block A does not
    change the shape of the input.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = F.relu(x)
        return x


class BlockB(nn.Module):
    """
    Block B is a residual block with a shortcut connection. Unlike Block A it
    halves size of the input. It is exclusively used in the encoder.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.bn3(self.conv3(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = F.relu(x)
        return x


class BlockC(nn.Module):
    """
    Block C is a residual block with a shortcut connection. It doubles size of
    the input. It is used exclusively in the decoder
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_sample(x)
        shortcut = self.bn3(self.conv3(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = F.relu(x)
        return x


class ResNet18Encoder(Encoder):
    """
    The Encoder component of an Auto Encoder. It is based on ResNet18
    architecture. It is used to encode the input image into a latent space.
    """

    def __init__(self, z_dim: int, data_shape: tuple = (3, 32, 32)):
        super().__init__()

        self.conv_00 = nn.Conv2d(
            data_shape[0], 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn_00 = nn.BatchNorm2d(64)
        self.block_01 = BlockA(64, 64)
        self.block_02 = BlockA(64, 64)
        self.block_03 = BlockB(64, 128)
        self.block_04 = BlockA(128, 128)
        self.block_05 = BlockB(128, 256)
        self.block_06 = BlockA(256, 256)
        self.block_07 = BlockB(256, 512)
        self.block_08 = BlockA(512, 512)

        self.fc = nn.Linear(512, z_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn_00(self.conv_00(x)))
        x = self.block_01(x)
        x = self.block_02(x)
        x = self.block_03(x)
        x = self.block_04(x)
        x = self.block_05(x)
        x = self.block_06(x)
        x = self.block_07(x)
        x = self.block_08(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ResNet18Decoder(Decoder):
    """
    The Decoder component of an Auto Encoder. It is based on ResNet18
    architecture. It is used to decode the latent space into an image.
    """

    def __init__(self, z_dim: int, data_shape: tuple = (3, 32, 32)):
        super().__init__()
        self.data_shape = data_shape

        self.fc = nn.Linear(z_dim, 512)

        self.block_01 = BlockA(512, 512)
        self.block_02 = BlockC(512, 256)
        self.block_03 = BlockA(256, 256)
        self.block_04 = BlockC(256, 128)
        self.block_05 = BlockA(128, 128)
        self.block_06 = BlockC(128, 64)
        self.block_07 = BlockA(64, 64)
        self.block_08 = BlockA(64, 64)

        self.conv_09 = nn.Conv2d(
            64, data_shape[0], kernel_size=3, padding=1, bias=False
        )
        self.bn_09 = nn.BatchNorm2d(data_shape[0])

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=self.data_shape[1] // 8)
        x = self.block_01(x)
        x = self.block_02(x)
        x = self.block_03(x)
        x = self.block_04(x)
        x = self.block_05(x)
        x = self.block_06(x)
        x = self.block_07(x)
        x = self.block_08(x)
        x = self.bn_09(self.conv_09(x))
        return torch.sigmoid(x)

    def decode(self, embedding: Tensor) -> Tensor:
        return self.forward(embedding)
