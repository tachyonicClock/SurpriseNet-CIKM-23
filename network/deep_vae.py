from avalanche.core import SupervisedPlugin
from experiment.strategy import ForwardOutput, Strategy
from network.hvae.oodd.losses import ELBO
from network.mlp import ClassifierHead
from .trait import AutoEncoder, Classifier, Encoder, Decoder, VariationalAutoEncoder, Samplable, MultiOutputNetwork
from network.hvae.oodd.layers.stages import VaeStage, LvaeStage
from network.hvae.oodd.models.dvae import DeepVAE
from network.hvae.oodd.variational import DeterministicWarmup, FreeNatsCooldown
from torch import nn, Tensor
import torch
from experiment.loss import LossObjective
import avalanche as av
import typing as t
import numpy as np


class HVAE(Classifier, Encoder, Decoder, Samplable, MultiOutputNetwork):

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.hvae = self.make_hvae()
        """An underlying hierarchical variational autoencoder"""

        final_latent_features = self.hvae.config_stochastic[-1]["latent_features"]
        self.classifier = self.make_classifier(final_latent_features, n_classes)
        """A classifier that takes the latent code as input"""
        
    
    def make_hvae(self) -> DeepVAE:
        raise NotImplementedError("HVAE is not implemented")
    
    def make_classifier(self, latent_features: int, n_classes: int) -> nn.Module:
        raise NotImplementedError("HVAE is not implemented")
        
    def forward(self, 
            x: Tensor,
            n_posterior_samples: int = 1,
            use_mode: bool | t.List[bool] = False,
            decode_from_p: bool | t.List[bool] = False,
            **stage_kwargs: t.Any        
        ):
        return self.hvae(x, n_posterior_samples, use_mode, decode_from_p, **stage_kwargs)

    def classify(self, x: Tensor) -> Tensor:
        return self.multi_forward(x).y_hat

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError("DeepVAE does not support encoding")

    def decode(self, posteriors: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError("DeepVAE does not support decoding")

    def sample(self, n: int) -> Tensor:
        likelihood_data, _ = self.hvae.sample_from_prior(n)
        samples: Tensor = likelihood_data.samples.view(n, *self.hvae.input_shape)
        return samples

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def multi_forward(self, x: Tensor) -> ForwardOutput:
        likelihood_data, stage_data = self.forward(
            x, n_posterior_samples=1)
        kl_divergences = [
            sd.loss.kl_elementwise for sd in stage_data if sd.loss.kl_elementwise is not None
        ]

        # Use the last stage's latent code as input to the classifier
        y_hat  = self.classifier(stage_data[-1].q.z)
        
        forward_output = ForwardOutput()
        forward_output.y_hat = y_hat
        forward_output.x_hat = likelihood_data.samples
        forward_output.likelihood = likelihood_data.likelihood
        forward_output.kl_divergences = kl_divergences
        return forward_output

    @property
    def n_latents(self) -> int:
        return self.hvae.n_latents

class FashionMNISTDeepVAE(HVAE):

    def make_hvae(self):
        stochastic_layers = [
            {"block": "GaussianConv2d", "latent_features": 8, "weightnorm": False},
            {"block": "GaussianDense", "latent_features": 16, "weightnorm": False},
            {"block": "GaussianDense", "latent_features": 8, "weightnorm": False}
        ]

        deterministic_layers = [
            [
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5,
                    "stride": 1, "weightnorm": False, "gated": False},
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5,
                    "stride": 1, "weightnorm": False, "gated": False},
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5,
                    "stride": 2, "weightnorm": False, "gated": False}
            ],
            [
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3,
                    "stride": 1, "weightnorm": False, "gated": False},
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3,
                    "stride": 1, "weightnorm": False, "gated": False},
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3,
                    "stride": 2, "weightnorm": False, "gated": False}
            ],
            [
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3,
                    "stride": 1, "weightnorm": False, "gated": False},
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3,
                    "stride": 1, "weightnorm": False, "gated": False},
                {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3,
                    "stride": 1, "weightnorm": False, "gated": False}
            ]
        ]

        return DeepVAE(
            Stage=VaeStage,
            input_shape=torch.Size([1, 32, 32]),
            likelihood_module='DiscretizedLogisticLikelihoodConv2d',
            config_deterministic=deterministic_layers,
            config_stochastic=stochastic_layers,
            q_dropout=0.0,
            p_dropout=0.0,
            activation='ReLU',
            skip_stochastic=True,
            padded_shape=None
        )
    
    def make_classifier(self, latent_features: int, n_classes: int) -> nn.Module:
        return ClassifierHead(latent_features, n_classes)


class Average():
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0
    
    def update(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)

    def get(self) -> float:
        return self.sum / self.count
    
    def reset(self) -> None:
        self.count = 0
        self.sum = 0


class DeepVAELoss(LossObjective, SupervisedPlugin):
    name = "DeepVAE"

    def __init__(self, 
            weighting: float,
            logger:  av.logging.TensorboardLogger,
            free_nat_start_value: float = None,
            free_nats_epochs: float = None,
            warmup_epochs: int = None,
            enable_free_nats: bool = False
            ) -> None:
        """Wrapper for ELBO loss for use with DeepVAE

        :param weighting: How important is this loss compared to others
        :param free_nat_start_value: Initial nats considered free in the KL term
        :param free_nats_epochs: Epochs to warm up the KL term
        :param warmup_epochs: Epochs to warm up the KL term
        """        
        
        super().__init__(weighting)
        self.elbo = ELBO()
        self.free_nat_start_value = free_nat_start_value
        self.free_nats_epochs = free_nats_epochs
        self.enable_free_nats = enable_free_nats
        self.warmup_epochs = warmup_epochs
        self.logger = logger.writer

        self.bpd = Average()
        self.elbo_metric = Average()
        self.loss_metric = Average()
        self.likelihood  = Average()
        self.kls: t.List[Average] = []

        self.before_training_exp(None)
        self.before_training_epoch(None)

    def before_training_exp(self, strategy: 'Template', *args, **kwargs):
        print("DeepVAE Loss: Initializing warmup and free nats")
        # Reset the warmup and free nats
        self.deterministic_warmup = DeterministicWarmup(n=self.warmup_epochs)
        if self.enable_free_nats:
            self.free_nats_cool_down = FreeNatsCooldown(
                constant_epochs=self.free_nats_epochs // 2,
                cooldown_epochs=self.free_nats_epochs // 2,
                start_val=self.free_nat_start_value,
                end_val=0,
            )

    def before_training_epoch(self, strategy: 'Template', *args, **kwargs):
        # Increment the simulated annealing
        self.beta = next(self.deterministic_warmup)
        if self.enable_free_nats:
            self.free_nats = next(self.free_nats_cool_down)
        self.bpd.reset()


    def before_eval(self, strategy, *args, **kwargs):
        self.t = 0
        self.bpd.reset()
        self.elbo_metric.reset()
        self.loss_metric.reset()
        self.likelihood.reset()
        for kl in self.kls:
            kl.reset()

    def after_training_epoch(self, strategy: Strategy, *args, **kwargs):
        self.logger.add_scalar("Train.hyperparameters/beta", self.beta, strategy.clock.train_iterations)
        self.logger.add_scalar(f"Train.likelihoods/log p(x)", self.elbo_metric.get(), strategy.clock.train_iterations)
        self.logger.add_scalar(f"Train.likelihoods/loss", self.loss_metric.get(), strategy.clock.train_iterations)
        self.logger.add_scalar(f"Train.likelihoods/log p(x|z)", self.likelihood.get(), strategy.clock.train_iterations)
        self.logger.add_scalar(f"Train.likelihoods/bpd", self.bpd.get(), strategy.clock.train_iterations)

        if self.enable_free_nats:
            self.logger.add_scalar("Train.hyperparameters/free_nats", self.free_nats, strategy.clock.train_iterations)
        for i, kl in enumerate(self.kls):
            self.logger.add_scalar(f"Train.divergences/kl_z{i}", kl.get(), strategy.clock.train_iterations)


    def after_eval_exp(self, strategy: Strategy, *args, **kwargs):
        self.logger.add_scalar(f"BitsPerDim/task={self.t}", self.bpd.get(), strategy.clock.total_iterations)
        self.t += 1

    def update(self, out: ForwardOutput, target: Tensor = None):
        # Calculate the loss and store the likelihood and kl_divergences
        assert out.likelihood is not None, "Expected likelihood to be provided"
        assert out.kl_divergences is not None, "Expected kl_divergences to be provided"

        loss, elbo, likelihood, kl_divergences = self.elbo.forward(
            out.likelihood,
            out.kl_divergences,
            free_nats=self.free_nats if self.enable_free_nats else 0.,
            beta=self.beta,
            sample_reduction=torch.mean,
            batch_reduction=None
        )
        self.loss = loss.mean()

        bpd: Tensor = - elbo.mean() / np.log(2.) / np.prod(out.x.shape[1:])
        self.bpd.update(bpd.item())
        self.elbo_metric.update(elbo.mean().item())
        self.loss_metric.update(loss.mean().item())
        self.likelihood.update(likelihood.mean().item())

        for i, kl in enumerate(out.kl_divergences):
            if len(self.kls) <= i:
                self.kls.append(Average())
            self.kls[i].update(kl.mean().item())

