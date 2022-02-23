import typing
from torch import Tensor, nn
from .trait import IsGenerative, TaskAware
import torchvision.transforms as transforms


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
            # 8x8 => 8x8
            nn.Conv2d(
                2 * c_hid, 2 * c_hid, 
                kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3,
                      padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
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
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


def MLP_Encoder(latent_dims):
    """3 Layer multi-layer perceptron encoder"""
    act_fn = nn.ReLU

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        act_fn(),
        nn.Linear(512, 256),
        act_fn(),
        nn.Linear(256, 128),
        act_fn(),
        nn.Linear(128, latent_dims),
    )


def MLP_Decoder(latent_dims):
    """3 Layer multi-layer perceptron decoder"""
    act_fn = nn.ReLU

    return nn.Sequential(
        nn.Linear(latent_dims, 128),
        act_fn(),
        nn.Linear(128, 256),
        act_fn(),
        nn.Linear(256, 512),
        act_fn(),
        nn.Linear(512, 28*28),
        nn.Tanh(),  # Scale output between -1 and +1
        nn.Unflatten(1, (1, 28, 28)),
    )


def MLP_AE_Head(latent_dims, output_size):
    return nn.Sequential(
        nn.Linear(latent_dims, output_size),
        nn.ReLU()
    )


class D_AE(nn.Module, IsGenerative):
    # Discriminative auto-encoder

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 head: nn.Module) -> None:
        super().__init__()

        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder
        self.head: nn.Module = head

    # Overload trait.IsGenerative functions
    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def forward(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        """ Forward through the neural network
        Returns:
            typing.Tuple[Tensor, Tensor]: Reconstruction and predicted class
                labels
        """
        z = self.encoder(x)      # Latent codes
        x_hat = self.decoder(z)  # Reconstruction
        y_hat = self.head(z)     # Classification head
        return x_hat, y_hat


class AEGroup(nn.Module, IsGenerative, TaskAware):
    """A group of autoencoders"""
    current_task: int = 0
    encoders: typing.Dict[int, nn.Module] = {}

    def __init__(self, make_auto_encoder: typing.Callable[[], nn.Module]):
        self.encoders[0] = make_auto_encoder()
        self.make_auto_encoder = make_auto_encoder

    def on_task_change(self, new_task_id: int):
        self.current_task = new_task_id

        if new_task_id not in self.encoders.keys():
            self.encoders[new_task_id] = self.make_auto_encoder()

    def forward(self, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        # If training we know the task
        if self.training:
            return self.get_encoder()(x)

        # TODO use task with lowest reconstruction loss
        raise NotImplemented("WIP ae_group:85")

    def get_encoder(self):
        return self.encoders[self.current_task]
