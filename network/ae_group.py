import typing
from torch import Tensor, nn
from .trait import IsGenerative, TaskAware


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
                 latent_dims: int,
                 output_layer_size: int,
                 encoder=MLP_Encoder,
                 decoder=MLP_Decoder,
                 head=MLP_AE_Head) -> None:
        super().__init__()

        self.encoder: nn.Module = encoder(latent_dims)
        self.decoder: nn.Module = decoder(latent_dims)
        self.head: nn.Module = head(latent_dims, output_layer_size)

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
