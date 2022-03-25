"""
`coders.py` contains encoder and decoder architectures
"""
import math
import typing
from torch import Tensor, nn
import torch
import network.module.packnet as pn

from network.trait import PackNet, PackNetParent


class CNN_Encoder(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. 
            For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first 
            convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            # 32x32 => 16x16
            nn.Conv2d(num_input_channels, c_hid,
                      kernel_size=3, padding=1, stride=2),
            act_fn(),
            # 16x16 => 16x16
            nn.Conv2d(c_hid, c_hid,
                      kernel_size=3, padding=1, stride=1),
            act_fn(),
            # 16x16 => 8x8
            nn.Conv2d(c_hid, 2 * c_hid,
                      kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(
                2 * c_hid, 2 * c_hid,
                kernel_size=3, padding=1),
            act_fn(),
            # 8x8 => 4x4
            nn.Conv2d(2 * c_hid, 2 * c_hid,
                      kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CNN_Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. 
            For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last 
            convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            # 4x4 => 8x8
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid,
                               kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid,
                      kernel_size=3, padding=1),
            act_fn(),
            # 8x8 => 16x16
            nn.ConvTranspose2d(2 * c_hid, c_hid,
                               kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid,
                      kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels,
                               kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x * 4 # Since we are normalizing data with a normal distribution this makes it so that Tanh is within the full range


class PN_CNN_Encoder(CNN_Encoder, PackNetParent):
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        super().__init__(
            num_input_channels, base_channel_size, latent_dim, act_fn)
        self.net = pn.wrap(self.net)


class PN_CNN_Decoder(CNN_Decoder, PackNetParent):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.net = pn.wrap(self.net)
        self.linear = pn.wrap(self.linear)


def _new_layer(
    input_features: int,
    output_features: int,
    linear: typing.Type[nn.Module] = nn.Linear,
    act_fn: typing.Type[nn.Module] = nn.ReLU,
):
    return nn.Sequential(
        linear(input_features, output_features),
        act_fn()
    )


class DenseEncoder(nn.Module):
    def __init__(self,
                 pattern_shape: torch.Size,
                 latent_dim: int,
                 layer_sizes: typing.Sequence[int],
                 linear: typing.Type[nn.Module] = nn.Linear,
                 act_fn: typing.Type[nn.Module] = nn.ReLU
                 ) -> None:
        super().__init__()
        def new_layer(x, y): return _new_layer(x, y, linear, act_fn)
        input_size = math.prod(pattern_shape)

        self.net = nn.Sequential(
            new_layer(input_size, layer_sizes[0]),
            *[new_layer(i_feat, o_feat)
                for i_feat, o_feat in zip(layer_sizes, layer_sizes[1:])],
            linear(layer_sizes[-1], latent_dim),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> Tensor:
        x = torch.flatten(input, 1)
        x = self.net(x)
        return x


class DenseDecoder(nn.Module):

    def __init__(self,
                 pattern_shape: torch.Size,
                 latent_dim: int,
                 layer_sizes: typing.Sequence[int],
                 linear: typing.Type[nn.Module] = nn.Linear,
                 act_fn: typing.Type[nn.Module] = nn.ReLU
                 ) -> None:
        super().__init__()

        def new_layer(x, y): return _new_layer(x, y, linear, act_fn)
        output_size = math.prod(pattern_shape)

        self.net = nn.Sequential(
            new_layer(latent_dim, layer_sizes[0]),
            *[new_layer(i_feat, o_feat)
                for i_feat, o_feat in zip(layer_sizes, layer_sizes[1:])],
            linear(layer_sizes[-1], output_size),
        )
        self.pattern_shape = pattern_shape

    def forward(self, input: Tensor) -> Tensor:
        x: Tensor = torch.flatten(input, 1)
        x = self.net(x)
        x = x.view(-1, *self.pattern_shape)
        return x


class PackNetDenseEncoder(DenseEncoder, PackNetParent):
    def __init__(self,
                 pattern_shape: torch.Size,
                 latent_dim: int,
                 layer_sizes: typing.Sequence[int]
                 ) -> None:
        super().__init__(pattern_shape, latent_dim, layer_sizes, pn.deffer_wrap(nn.Linear))


class PackNetDenseDecoder(DenseDecoder, PackNetParent):
    def __init__(self,
                 pattern_shape: torch.Size,
                 latent_dim: int,
                 layer_sizes: typing.Sequence[int]
                 ) -> None:

        super().__init__(pattern_shape, latent_dim, layer_sizes, pn.deffer_wrap(nn.Linear))


class PackNetDenseHead(PackNetParent):
    def __init__(self, latent_dims, output_size):
        super().__init__()

        self.net = nn.Sequential(
            pn.wrap(nn.Linear(latent_dims, output_size)),
            nn.ReLU()
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)
