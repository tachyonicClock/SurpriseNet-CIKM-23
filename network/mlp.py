from numpy import outer
import torch
from torch import nn

from network.trait import Classifier, Decoder, Encoder


class MLPEncoder(Encoder):
    """
    MLP encoder.
    """

    def __init__(self, in_dimension, latent_dims):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimension, in_dimension*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_dimension*2, in_dimension*4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_dimension*4, latent_dims),
        )


    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        return self(x)

class MLPDecoder(Decoder):
    """
    MLP decoder.
    """

    def __init__(self, latent_dims, out_dimension):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dims, out_dimension*4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_dimension*4, out_dimension*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_dimension*2, out_dimension),
        )


    def forward(self, x):
        return self.net(x)

    def decode(self, x):
        return self(x)


class UniformMLPEncoder(Encoder):
    """
    MLP Encoder with roughly constant number of weights per layer
    """

    def __init__(self,
        in_dimensions: int,
        width: int,
        depth: int,
        latent_dims: int,
        dropout: float = 0.5
    ):
        super().__init__()
        assert depth >= 4, "At least four layers are needed"

        min_weight_count = width*width
        # The number of features in the first layer is chosen such that the
        # number of weights is not smaller than min_weight_count
        outer_input = max(min_weight_count//in_dimensions, width)
        outer_output = max(min_weight_count//latent_dims, width)

        print(f"Encoder outer_input: {outer_input}")
        print(f"Encoder outer_output: {outer_output}")
        

        self.outer_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dimensions, outer_input),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outer_input, width),
            nn.Dropout(dropout),
            nn.ReLU() 
        )

        self.inner_layers = nn.Sequential(
            *[nn.Sequential(
                    nn.Linear(width, width),
                    nn.Dropout(dropout),
                    nn.ReLU()
            ) for _ in range(depth-3)]
        )

        self.outer_layers_latent = nn.Sequential(
            nn.Linear(width, outer_output),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(outer_output, latent_dims),
        )

    def forward(self, x):
        x = self.outer_layers(x)
        x = self.inner_layers(x)
        x = self.outer_layers_latent(x)
        return x

    def encode(self, x):
        return self(x)


class UniformMLPDecoder(Decoder):
    """
    MLP Decoder with roughly constant number of weights per layer
    """

    def __init__(self,
        out_dimensions: int,
        width: int,
        depth: int,
        latent_dims: int,
        dropout: float = 0.5
    ):
        super().__init__()
        assert depth >= 4, "At least four layers are needed"
        # The number of features in the first layer and before the latent space
        # is chosen such that the number of weights is not smaller 
        # than min_weight_count
        min_weight_count = width*width
        outer_input = max(min_weight_count//out_dimensions, width)
        outer_output = max(min_weight_count//latent_dims, width)


        print(f"Decoder outer_input: {outer_input}")
        print(f"Decoder outer_output: {outer_output}")

        self.outer_layers_latent = nn.Sequential(
            nn.Linear(latent_dims, outer_output),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(outer_output, width),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.inner_layers = nn.Sequential(
            *[nn.Sequential(
                    nn.Linear(width, width),
                    nn.Dropout(dropout),
                    nn.ReLU()
            ) for _ in range(depth-3)]
        )

        self.outer_layers = nn.Sequential(
            nn.Linear(width, outer_input),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(outer_input, out_dimensions),
        )

    def forward(self, x):
        x = self.outer_layers_latent(x)
        x = self.inner_layers(x)
        x = self.outer_layers(x)
        return x

    def decode(self, x):
        return self(x)

class MPLRectangularClassifierHead(Classifier):
    """
    MLP Classifier with roughly constant width/number of weights
    """

    def __init__(self,
        latent_dims: int,
        width: int,
        num_classes: int
    ):
        super().__init__()
        # The number of features in the first layer and before the latent space
        # is chosen such that the number of weights is not smaller
        # than min_weight_count
        min_weight_count = width*width
        classifier_features = max(min_weight_count//latent_dims, min_weight_count//num_classes)

        self.net = nn.Sequential(
            nn.Linear(latent_dims, classifier_features),
            nn.ReLU(),
            nn.Linear(classifier_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def classify(self, embedding) -> torch.Tensor:
        return self.net(embedding)