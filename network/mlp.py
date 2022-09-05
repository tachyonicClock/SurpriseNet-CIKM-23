import typing as t
from torch import nn, Tensor, Size
from network.trait import Classifier, Decoder, Encoder, Sampler


def _mlp_layer(in_features: int, out_features: int, dropout: float):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Dropout(dropout),
        nn.ReLU()
    )


class MLPEncoder(Encoder):
    """
    The encoder component of an Auto Encoder. Turns an image into a latent vector.
    """

    def __init__(self, z_dim: int, data_shape: tuple, width: int = 512, dropout: float = 0.5):
        super().__init__()
        total_features = sum(data_shape)

        self.layers = nn.Sequential(
            nn.Flatten(),
            _mlp_layer(total_features, width*4, 0.0),
            _mlp_layer(width*4, width*2, dropout),
            _mlp_layer(width*2, width, dropout),
            nn.Linear(width, z_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def encode(self, x: Tensor) -> Tensor:
        return self(x)


class MLPDecoder(Decoder):
    """
    A decoder component of an auto encoder. Turns a latent vector into an image.
    """

    def __init__(self, z_dim: int, data_shape: tuple, width: int = 512, dropout: float = 0.5):
        super().__init__()
        total_features = sum(data_shape)

        self.layers = nn.Sequential(
            _mlp_layer(z_dim, width, 0.0),
            _mlp_layer(width, width*2, dropout),
            _mlp_layer(width*2, width*4, dropout),
            nn.Linear(width*4, total_features),
            nn.Unflatten(1, data_shape),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def decode(self, x):
        return self(x)


class ClassifierHead(Classifier):
    """
    The classifier head turns a latent vector into a class label.
    """

    def __init__(self, z_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim*4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(z_dim*4, n_classes),
        )

    def classify(self, embedding: Tensor) -> Tensor:
        return self(embedding)

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.net(x)
        return y_hat


class VAEBottleneck(Sampler):
    """
    The bottleneck of a VAE. Turns a latent vector into a mean and variance.
    """

    def __init__(self, in_features: int, z_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(in_features, z_dim, bias=False)
        self.log_var = nn.Linear(in_features, z_dim, bias=False)
        self.z_dim = z_dim

    @property
    def bottleneck_width(self) -> int:
        return self.z_dim

    def forward(self, x: Tensor) -> t.Tuple[Tensor, Tensor]:
        return self.mu(x), self.log_var(x)
