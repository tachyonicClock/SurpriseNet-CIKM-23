from torch import nn, Tensor
from network.trait import Decoder, PackNetComposite
import network.module.packnet as pn

class CNN_Decoder(Decoder, nn.Module):
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
        self.latent_dim = latent_dim
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            act_fn(),
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn(),
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
        return x * 4 # *4 because Tanh is between -1 and 1 but the data
                     # follows a normal distribution. x4 contains 99.99% of the
                     # range

    def decode(self, z: Tensor) -> Tensor:
        return self.forward(z)
    
    @property
    def bottleneck_width(self) -> int:
        return self.latent_dim

class PN_CNN_Decoder(CNN_Decoder, PackNetComposite):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.net = pn.wrap(self.net)
        self.linear = pn.wrap(self.linear)