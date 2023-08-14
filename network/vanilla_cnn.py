import torch
import typing as t
from torch import nn, Tensor
from .trait import Decoder, Encoder


class ConvUpSample(nn.Module):
    """
    ConvUpSample doubles the size of the input
    """

    def __init__(
        self, in_channels: int, out_channels: int, act_fn=nn.ReLU(), batch_norm=True
    ):
        super().__init__()
        # Odena, et al. recommends up sample and then conv2d to avoid
        # checkerboard artifacts in reconstructions
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1
        )
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act_fn = act_fn

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_sample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x


class ConvDownSample(nn.Module):
    """
    Halves the size of the input
    """

    def __init__(
        self, in_channels: int, out_channels: int, act_fn=nn.ReLU(), batch_norm=True
    ):
        super().__init__()
        self.act_fn = act_fn
        self.conv_down_sample = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=2
        )
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_down_sample(x)
        x = self.bn(x)
        x = self.act_fn(x)
        return x


class VanillaCNNEncoder(Encoder):
    """
    A five layer Encoder. Scales an image down by a factor of 2^3 before using a
    linear layer to map it to latent code
    """

    def __init__(self, z_dim: int, data_shape: tuple, base_channels: int):
        """Create the Encoder part of a CNN Auto Encoder"""
        super().__init__()
        nc = data_shape[0]

        # (B, 3, 32, 32)
        self.conv_01 = ConvDownSample(nc, base_channels)
        # (B, 128, 16, 16)
        self.conv_02 = ConvDownSample(base_channels, base_channels * 2)
        # (B, 256, 8, 8)
        self.conv_03 = ConvDownSample(base_channels * 2, base_channels * 4)
        # (B, 512, 4, 4)
        self.conv_04 = ConvDownSample(base_channels * 4, base_channels * 8)
        # (B, 1024, 2, 2)
        self.fc = nn.Linear(base_channels * 8 * 2 * 2, z_dim)

    def forward(self, pattern: Tensor) -> Tensor:
        x: Tensor = self.conv_01(pattern)
        x = self.conv_02(x)
        x = self.conv_03(x)
        x = self.conv_04(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)

    def encode(self, pattern: Tensor) -> Tensor:
        return self(pattern)


class VanillaCNNDecoder(Decoder):
    """
    A five layer decoder. Up samples an embedding to 32x32
    """

    def __init__(self, z_dim: int, data_shape: tuple, base_channels: int):
        """Create the Decoder part of a CNN Auto Encoder"""
        super().__init__()
        nc = data_shape[0]
        self.fc = nn.Linear(z_dim, base_channels * 8 * 4)

        # (B, 1024, 2, 2)
        self.resize_conv_01 = ConvUpSample(base_channels * 8, base_channels * 4)
        # (B, 512, 4, 4)
        self.resize_conv_02 = ConvUpSample(base_channels * 4, base_channels * 2)
        # (B, 256, 8, 8)
        self.resize_conv_03 = ConvUpSample(base_channels * 2, base_channels)
        # (B, 128, 16, 16)
        self.resize_conv_04 = ConvUpSample(base_channels, base_channels)
        # (B, 3, 32, 32)

        self.final_conv = nn.Conv2d(
            base_channels, nc, kernel_size=3, padding=1, stride=1
        )

        self.base_dim = base_channels

    def forward(self, embedding: Tensor) -> Tensor:
        x: Tensor = self.fc(embedding)
        x = x.reshape(x.shape[0], -1, 2, 2)

        x = self.resize_conv_01(x)
        x = self.resize_conv_02(x)
        x = self.resize_conv_03(x)
        x = self.resize_conv_04(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)

    def decode(self, embedding: Tensor) -> Tensor:
        return self(embedding)
