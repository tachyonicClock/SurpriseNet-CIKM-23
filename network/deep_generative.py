from dataclasses import dataclass
from turtle import forward
import typing
from sqlalchemy import false
from torch import Tensor, nn
import torch
from torch.nn import functional as F

from experiment.strategy import ForwardOutput
from functional import MRAE
from functional.task_inference import infer_task
from .trait import Classifier, ConditionedSample, Decoder, InferTask, PackNet, PackNetComposite, Sampler, Samplable, Encoder, AutoEncoder


class DAE(AutoEncoder, Classifier, nn.Module):
    """
    Discriminative Auto-Encoder
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        classifier: Classifier) -> None:
        super().__init__()

        assert encoder.bottleneck_width == decoder.bottleneck_width, \
            f"The encoder({encoder.bottleneck_width}) and decoder must have the same bottleneck width ({decoder.bottleneck_width})"

        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder.encode(x)
    
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder.decode(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    @property
    def bottleneck_width(self) -> int:
        return self.encoder.bottleneck_width

    def forward(self, x: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        out.z_code = self.encoder.encode(x)
        out.x_hat = self.decoder.decode(out.z_code)
        out.y_hat = self.classifier.classify(out.z_code)
        out.x = x
        return out

class PN_DAE(PackNetComposite, DAE):
    def __init__(self,
        encoder: typing.Union[Encoder, PackNet],
        decoder: typing.Union[Decoder, PackNet],
        classifier: typing.Union[Classifier, PackNet]) -> None:
        super().__init__(encoder, decoder, classifier)

class PN_DAE_InferTask(InferTask, PN_DAE):

    subnet_count: int = 0

    def forward(self, x: Tensor) -> ForwardOutput:
        if self.training:
            return super().forward(x)
        return infer_task(super(), x, self.subnet_count, 1)

    def push_pruned(self):
        super().push_pruned()
        self.subnet_count += 1

class DVAE(AutoEncoder, Classifier, Samplable, nn.Module):

    def __init__(self,
        encoder: Encoder,
        probabilistic_encoder: Sampler,
        decoder: Decoder,
        classifier: Classifier) -> None:
        super().__init__()

        self.dummy_param = nn.Parameter(torch.empty(0)) # Used to determine device
        self.encoder = encoder
        self.probabilistic_encoder = probabilistic_encoder
        self.decoder = decoder
        self.classifier = classifier

    def encode(self, x: Tensor) -> Tensor:
        mu, std = self.probabilistic_encoder.encode(self.encoder.encode(x))
        return self.probabilistic_encoder.reparameterise(mu, std)
    
    def sample(self, n: int = 1) -> Tensor:
        return self.decode(self.probabilistic_encoder.sample(n).to(self.dummy_param.device))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder.decode(z)

    def classify(self, x: Tensor) -> Tensor:
        return self.forward(x).y_hat

    @property
    def bottleneck_width(self) -> int:
        return self.probabilistic_encoder.bottleneck_width

    def forward(self, x: Tensor) -> ForwardOutput:
        out = ForwardOutput()
        z = self.encoder.encode(x)
        out.mu, out.log_var = self.probabilistic_encoder.encode(z)
        out.z_code = self.probabilistic_encoder.reparameterise(out.mu, out.log_var)
        out.y_hat = self.classifier(out.z_code)
        out.x_hat = self.decoder(out.z_code)
        out.x = x
        return out

    def sample_x(self, n:int=1) -> Tensor:
        return torch.randn(n, self.latent_dim)

class PN_DVAE(PackNetComposite, ConditionedSample, DVAE):


    def __init__(self,
        encoder: typing.Union[Encoder, PackNet],
        probabilistic_encoder: typing.Union[Sampler, PackNet],
        decoder: typing.Union[Decoder, PackNet],
        classifier: typing.Union[Classifier, PackNet]) -> None:
        super().__init__(encoder, probabilistic_encoder, decoder, classifier)

    def conditioned_sample(self, n: int = 1, given_task: int = 0) -> Tensor:
        self.use_task_subset(given_task)
        sample = self.sample(n)
        self.use_top_subset()
        return sample


class PN_DVAE_InferTask(InferTask, PN_DVAE):

    subnet_count: int = 0

    def forward(self, x: Tensor) -> ForwardOutput:
        if self.training:
            return super().forward(x)
        return infer_task(super(), x, self.subnet_count, 30)

    def push_pruned(self):
        super().push_pruned()
        self.subnet_count += 1

