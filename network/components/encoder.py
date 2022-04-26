from torch import nn, Tensor
from network.trait import Encoder, PackNetComposite
import network.module.packnet as pn

class CNN_Encoder(Encoder, nn.Module):
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
        self.latent_dim = latent_dim
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
            # 8x8 => 8x8
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
            act_fn(),
            nn.Linear(latent_dim, latent_dim),
            act_fn(),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def bottleneck_width(self) -> int:
        return self.latent_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.forward(x)


class PN_CNN_Encoder(CNN_Encoder, PackNetComposite):
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        super().__init__(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.net = pn.wrap(self.net)
