from torch import nn, Tensor
from network.trait import Encoder, PackNetComposite
import network.module.packnet as pn

class CNN_Encoder(Encoder, nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 kernel_number: int,
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
        self.act_fn = act_fn
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # 32x32
            self._conv(num_input_channels, kernel_number//4),
            # 16x16
            self._conv(kernel_number // 4, kernel_number//2),
            # 8x8
            self._conv(kernel_number // 2, kernel_number),
            # 4x4
            nn.Flatten(),
            nn.Linear(kernel_number*16, latent_dim),
            act_fn(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def bottleneck_width(self) -> int:
        return self.latent_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def _conv(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            self.act_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            self.act_fn(),
        )



class PN_CNN_Encoder(CNN_Encoder, PackNetComposite):
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        super().__init__(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.net = pn.wrap(self.net)



