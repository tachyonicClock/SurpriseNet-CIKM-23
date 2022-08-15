import torch
import typing as t
from torch import nn, Tensor, tensor
from .trait import Classifier, Decoder, Encoder, Sampler


class VanillaCNNEncoder(Encoder):
    """
    A 5 layer Encoder for CNN Auto Encoder. Scales an image down by a factor of
    2^3 before using a linear layer to map it to a latent code
    """

    def __init__(self,
                 in_channels: int,
                 base_dim: int,
                 latent_dim: int):
        """Create the Encoder part of a CNN Auto Encoder

        :param in_channels: The number of channels in the input (3 for color)
        :param base_dim:    Doubled in each successive layer
        :param latent_dim:  The number of latent variables to output
        """
        super().__init__()
        self.act_fn = nn.ReLU(True)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, stride=2),
                self.act_fn
            )

        # (B, 3, 32, 32)
        self.conv_01 = conv(in_channels, base_dim)
        # (B, 128, 16, 16)
        self.conv_02 = conv(base_dim, base_dim*2)
        # (B, 256, 8, 8)
        self.conv_03 = conv(base_dim*2, base_dim*4)
        # (B, 512, 4, 4)
        self.conv_04 = conv(base_dim*4, base_dim*8)
        # (B, 1024, 2, 2)
        self.fc = nn.Linear(base_dim*8*2*2, latent_dim)

    def forward(self, observation: Tensor) -> Tensor:
        x: Tensor = self.conv_01(observation)
        x = self.conv_02(x)
        x = self.conv_03(x)
        x = self.conv_04(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    def encode(self, observations: Tensor) -> Tensor:
        return self(observations)


class VanillaCNNDecoder(Decoder):

    def __init__(self,
                 decoded_channels: int,
                 base_dim: int,
                 latent_dim: int):
        """Create the Decoder part of a CNN Auto Encoder

        :param decoded_channels: The number of channels that are output 
            (3 for color)
        :param base_dim: Starting at base_dim*2^4 the number of channels is
            halved on each successive layer. Such that it is symmetrical with
            the Encoder 
        :param latent_dim:  The number of latent variables accepted as an input
        """
        super().__init__()
        self.act_fn = nn.ReLU(False)

        def de_conv(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
                self.act_fn
            )

        self.fc = nn.Linear(latent_dim, base_dim*8*4)

        # (B, 1024, 2, 2)
        self.de_conv_01 = de_conv(base_dim*8, base_dim*4)
        # (B, 512, 4, 4)
        self.de_conv_02 = de_conv(base_dim*4, base_dim*2)
        # (B, 256, 8, 8)
        self.de_conv_03 = de_conv(base_dim*2, base_dim)
        # (B, 128, 16, 16)
        self.de_conv_04 = nn.ConvTranspose2d(
            base_dim, decoded_channels, kernel_size=3, output_padding=1, padding=1, stride=2)
        # (B, 3, 32, 32)

        self.base_dim = base_dim
        self.decoded_channels = decoded_channels

    def forward(self, embedding: Tensor) -> Tensor:
        x: Tensor = self.fc(embedding)
        x = x.reshape(x.shape[0], -1, 2, 2)

        x = self.de_conv_01(x)
        x = self.de_conv_02(x)
        x = self.de_conv_03(x)
        x = self.de_conv_04(x)

        return torch.sigmoid(x)

    def decode(self, embedding: Tensor) -> Tensor:
        return self(embedding)


class ClassifierHead(Classifier):

    def __init__(self, latent_dims: int, class_number: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dims, latent_dims*4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(latent_dims*4, class_number),
        )

    def classify(self, embedding: Tensor) -> Tensor:
        return self(embedding)

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.net(x)
        return y_hat

class VAEBottleneck(Sampler):

    def __init__(self, input_width: int, latent_dims: int) -> None:
        super().__init__()
        self.mu = nn.Linear(input_width, latent_dims, bias=False)
        self.log_var = nn.Linear(input_width, latent_dims, bias=False)
        self.latent_dims = latent_dims

    @property
    def bottleneck_width(self) -> int:
        return self.latent_dims

    def forward(self, x: Tensor) -> t.Tuple[Tensor, Tensor]:
        return self.mu(x), self.log_var(x)
    