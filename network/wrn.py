from turtle import forward
import typing as t
from torch import Tensor, nn
import torch
from network.trait import Decoder, Encoder


class BasicBlock(nn.Module):
    """
    The basic block of a wide residual network
    https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 drop_rate: float = 0.0,
                 batch_norm_module: t.Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.in_channels = in_channels

        self.bn_0 = batch_norm_module(in_channels)
        self.relu_0 = nn.ReLU(True)
        self.conv_0 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)

        self.bn_1 = batch_norm_module(out_channels)
        self.relu_1 = nn.ReLU(True)
        self.conv_1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.dropout = nn.Dropout(drop_rate)

        # If our block changes the shape of the input we must also change the
        # shape of the shortcut connection
        self.equal_in_out = ((in_channels == out_channels) and (stride == 1))
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.in_channels, \
            f"Expected {self.in_channels} in channels got {x.shape[1]}"

        if self.equal_in_out:
            x_shortcut = x
            x = self.relu_0(self.bn_0(x))
        else:
            x = self.relu_0(self.bn_0(x))
            x_shortcut = self.shortcut(x)

        x = self.conv_0(x)

        x = self.relu_1(self.bn_1(x))
        x = self.conv_1(x)

        # Add residual connection
        return x + x_shortcut


class WidResNetBlock(nn.Module):
    """
    A WidResNetBlock made by stacking BasicBlocks
    """

    def __init__(self, depth,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 drop_rate: float = 0.0,
                 batch_norm_module: t.Type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()

        layers = []
        for i in range(depth):
            # The first layer is the only layer allowed to have a stride other
            # than 1. Alternatively, the first layer is the only layer allowed
            # to change the shape of the tensor
            if i == 0:
                layers.append(
                    BasicBlock(
                        in_channels,
                        out_channels,
                        stride,
                        drop_rate,
                        batch_norm_module))
            else:
                layers.append(
                    BasicBlock(
                        out_channels,
                        out_channels,
                        1,
                        drop_rate,
                        batch_norm_module))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class WideResNetEncoder(Encoder):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 widen_factor: int,
                 depth: int
                 ) -> None:
        super().__init__()

        assert depth % 3 == 0, "Depth must be multiple of 3"
        self.num_block_layers = int(depth / 3)
        wf = widen_factor
        self.conv_0 = nn.Conv2d(in_channels, 16*wf, 3, 1, 1, bias=False)
        self.encoder = nn.Sequential(
            WidResNetBlock(self.num_block_layers, wf*16, wf*32),
            WidResNetBlock(self.num_block_layers, wf*32, wf*64, stride=2),
            WidResNetBlock(self.num_block_layers, wf*64, wf*64, stride=2),
        )

        # hardcoded for 32x32 images
        self.fc = nn.Linear(8*8*wf*64, latent_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_0(x)
        x = self.encoder(x)
        x = self.fc(x.flatten(start_dim=1))
        return torch.sigmoid(x)

    def encode(self, observations: Tensor) -> Tensor:
        return self(observations)


class WideResNetDecoder(Decoder):

    def __init__(self,
                 output_channels: int,
                 latent_dim: int,
                 widen_factor: int,
                 depth: int
                 ) -> None:
        super().__init__()

        assert depth % 3 == 0, "Depth must be multiple of 3"
        self.num_block_layers = int(depth / 3)
        wf = widen_factor
        self.decoder = nn.Sequential(
            WidResNetBlock(self.num_block_layers, wf*64, wf*64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            WidResNetBlock(self.num_block_layers, wf*64, wf*32),
            nn.Upsample(scale_factor=2, mode='nearest'),
            WidResNetBlock(self.num_block_layers, wf*32, wf*16),
        )
        self.conv_last = nn.Conv2d(16*wf, output_channels, 3, 1, 1, bias=False)

        # hardcoded for 32x32 images
        self.fc = nn.Linear(latent_dim, 8*8*wf*64, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view((x.shape[0], -1, 8, 8))
        x = self.decoder(x)
        x = self.conv_last(x)
        return torch.sigmoid(x)

    def decode(self, embedding: Tensor) -> Tensor:
        return self(embedding)
