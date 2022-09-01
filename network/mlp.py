from numpy import outer
import torch
from torch import nn

from network.trait import Classifier, Decoder, Encoder

def _mlp_layer(in_features: int, out_features: int, dropout: float):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Dropout(dropout),
        nn.ReLU()
    )

class MLPEncoder(Encoder):
    """
    An MLP encoder with dropout and relu activation
    """

    def __init__(self,
        in_dimensions: int,
        width: int,
        latent_dims: int,
        dropout: float = 0.5
    ):
        super().__init__()

        self.layers = nn.Sequential(
            _mlp_layer(in_dimensions, width*4, 0.0),
            _mlp_layer(width*4, width*2, dropout),
            _mlp_layer(width*2, width, dropout),
            nn.Linear(width, latent_dims),
        )

    def forward(self, x):
        return self.layers(x)

    def encode(self, x):
        return self(x)


class MLPDecoder(Decoder):
    """
    An MLP decoder with dropout and relu activation
    """

    def __init__(self,
        out_dimensions: int,
        width: int,
        latent_dims: int,
        dropout: float = 0.5
    ):
        super().__init__()

        self.layers = nn.Sequential(
            _mlp_layer(latent_dims, width, 0.0),
            _mlp_layer(width, width*2, dropout),
            _mlp_layer(width*2, width*4, dropout),
            nn.Linear(width*4, out_dimensions),
        )

    def forward(self, x):
        return self.layers(x)

    def decode(self, x):
        return self(x)

class MLPClassifierHead(Classifier):
    """
    A simple MLP for classification
    """

    def __init__(self,
        latent_dims: int,
        width: int,
        num_classes: int
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dims, width*4),
            nn.ReLU(),
            nn.Linear(width*4, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def classify(self, embedding) -> torch.Tensor:
        return self.net(embedding)