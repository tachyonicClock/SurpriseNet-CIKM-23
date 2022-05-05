from torch import nn, Tensor, sigmoid
from network.trait import Decoder, PackNetComposite
import network.module.packnet as pn

class CNN_Decoder(Decoder, nn.Module):
    def __init__(self, num_input_channels: int, kernel_number: int, latent_dim: int, act_fn: object = nn.GELU):
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
        self.act_fn = act_fn
        self.latent_dim = latent_dim
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, kernel_number*16),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            # 4x4
            self._deconv(kernel_number, kernel_number // 2),
            # 8x8
            self._deconv(kernel_number//2, kernel_number // 4),
            # 16x16
            nn.ConvTranspose2d(kernel_number//4, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x

    def decode(self, z: Tensor) -> Tensor:
        return self.forward(z)
    
    @property
    def bottleneck_width(self) -> int:
        return self.latent_dim

    def _deconv(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GroupNorm(8, out_channels),
            self.act_fn(),
        )



class PN_CNN_Decoder(CNN_Decoder, PackNetComposite):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.net = pn.wrap(self.net)
        self.linear = pn.wrap(self.linear)