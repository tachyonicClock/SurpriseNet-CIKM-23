import math
import typing as t
from torch import nn, Tensor
from network.trait import Classifier, Decoder, Encoder, Sampler


def _mlp_layer(in_features: int, out_features: int, dropout: float):
    return nn.Sequential(
        nn.Linear(in_features, out_features), nn.Dropout(dropout), nn.ReLU()
    )


class MLPEncoder(Encoder):
    """
    MLPEncoder implements a multi-layer perceptron encoder for an autoencoder.
    """

    def __init__(
        self,
        latent_size: int,
        data_shape: tuple,
        width: int,
        layer_count: int,
        dropout: float,
        layer_growth: float = 2.0,
    ) -> None:
        """_summary_

        :param latent_size: The size of the latent representation.
        :param data_shape: The shape of the input data.
        :param width: The width of the hidden layers. Each hidden layer will have
        half the width of the previous layer.
        :param layer_count:  The number of hidden layers.
        :param dropout:  The dropout rate.
        """
        super().__init__()
        total_features = math.prod(data_shape)
        latent_size = int(latent_size)
        width = int(width)
        layer_count = int(layer_count)

        layers = [nn.Flatten()]
        features_in = total_features
        features_out = width
        for _ in range(layer_count - 1):
            layers.append(_mlp_layer(features_in, features_out, dropout))
            features_in = features_out
            features_out = int(features_out / layer_growth)

        layers.append(nn.Linear(features_in, latent_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def encode(self, x: Tensor) -> Tensor:
        return self(x)


class MLPDecoder(Decoder):
    """
    The decoder component of an Auto Encoder.
    """

    def __init__(
        self,
        latent_size: int,
        data_shape: tuple,
        width: int,
        layer_count: int,
        dropout: float,
        layer_growth: float = 2.0,
    ) -> None:
        super().__init__()
        latent_size = int(latent_size)
        width = int(width)
        layer_count = int(layer_count)
        total_features = math.prod(data_shape)

        layers = []
        features_in = latent_size
        features_out = int(width / layer_growth ** (layer_count - 2))
        for _ in range(layer_count - 1):
            layers.append(_mlp_layer(features_in, features_out, dropout))
            features_in = features_out
            features_out = int(features_out * layer_growth)

        layers.append(nn.Linear(features_in, total_features))
        self.layers = nn.Sequential(*layers)

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
            nn.Linear(z_dim, n_classes),
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
