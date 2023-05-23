import itertools
from hvae.hvaeoodd.oodd.layers.stages import BivaStage, VaeStage
from hvae.hvaeoodd.oodd.losses import ELBO
from hvae.hvaeoodd.oodd.models.dvae import DeepVAE
from hvae.hvaeoodd.oodd.variational.deterministic_warmup import (
    DeterministicWarmup,
)
from hvae.hvaeoodd.oodd.variational.free_nats import FreeNatsCooldown
from avalanche.core import SupervisedPlugin
from experiment.strategy import ForwardOutput, Strategy
from network.mlp import ClassifierHead
from network.trait import (
    Classifier,
    Encoder,
    Decoder,
    Samplable,
    MultiOutputNetwork,
)
from torch import nn, Tensor
import torch
from experiment.loss import LossObjective
import avalanche as av
import typing as t
import numpy as np
import copy


class HVAE(Classifier, Encoder, Decoder, Samplable, MultiOutputNetwork):
    def __init__(
        self, n_classes: int, latent_dims: int, base_channels: int, **kwargs
    ) -> None:
        """A hierarchical variational autoencoder.

        :param n_classes: The number of classes in the dataset
        :param latent_dims: The number of innermost latent dimensions
        """
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.hvae = self.make_hvae(
            latent_dims, base_channels=int(base_channels), **kwargs
        )
        """An underlying hierarchical variational autoencoder"""
        self.classifier = self.make_classifier(latent_dims, n_classes)
        """A classifier that takes the latent code as input"""

    def make_hvae(self, latent_dims: int, base_channels: int, **kwargs) -> DeepVAE:
        raise NotImplementedError("HVAE is not implemented")

    def make_classifier(self, latent_features: int, n_classes: int) -> nn.Module:
        raise NotImplementedError("HVAE is not implemented")

    def forward(
        self,
        x: Tensor,
        n_posterior_samples: int = 1,
        use_mode: bool | t.List[bool] = False,
        decode_from_p: bool | t.List[bool] = False,
        **stage_kwargs: t.Any,
    ):
        return self.hvae(
            x, n_posterior_samples, use_mode, decode_from_p, **stage_kwargs
        )

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
        likelihood_data, stage_data = self.forward(x, n_posterior_samples=1)
        kl_divergences = [
            sd.loss.kl_elementwise
            for sd in stage_data
            if sd.loss.kl_elementwise is not None
        ]

        # Use the last stage's latent code as input to the classifier
        y_hat = self.classifier(stage_data[-1].q.z)

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
    def make_hvae(
        self,
        latent_dims: int,
        base_channels: int,
        dropout: float = 0.0,
        stage_type: t.Literal["BivaStage", "VaeStage"] = "VaeStage",
    ) -> DeepVAE:
        stochastic_layers = [
            {"block": "GaussianConv2d", "latent_features": 8, "weightnorm": False},
            {"block": "GaussianDense", "latent_features": 16, "weightnorm": False},
            {
                "block": "GaussianDense",
                "latent_features": latent_dims,
                "weightnorm": False,
            },
        ]

        deterministic_layers = [
            [
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 5,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 5,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 5,
                    "stride": 2,
                    "weightnorm": False,
                    "gated": False,
                },
            ],
            [
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 3,
                    "stride": 2,
                    "weightnorm": False,
                    "gated": False,
                },
            ],
            [
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
                {
                    "block": "ResBlockConv2d",
                    "out_channels": base_channels,
                    "kernel_size": 3,
                    "stride": 1,
                    "weightnorm": False,
                    "gated": False,
                },
            ],
        ]

        _stage_types = {
            "BivaStage": BivaStage,
            "VaeStage": VaeStage,
        }

        return DeepVAE(
            Stage=_stage_types[stage_type],
            input_shape=torch.Size([1, 32, 32]),
            likelihood_module="DiscretizedLogisticLikelihoodConv2d",
            config_deterministic=deterministic_layers,
            config_stochastic=stochastic_layers,
            q_dropout=dropout,
            p_dropout=dropout,
            activation="ReLU",
            skip_stochastic=True,
            padded_shape=None,
        )

    def make_classifier(self, latent_features: int, n_classes: int) -> nn.Module:
        return ClassifierHead(latent_features, n_classes)


class Average:
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


class CyclicBeta(t.Iterable):
    def __init__(
        self,
        total_epochs: int,
        ratio: float = 0.5,
        cycles: int = 4,
    ) -> None:
        """
        Cyclic Annealing Schedule: A Simple Approach to Mitigating KL Vanishing


        Fu, H., Li, C., Liu, X., Gao, J., Celikyilmaz, A., & Carin, L. (2019).
        Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
        Vanishing (arXiv:1903.10145). arXiv. http://arxiv.org/abs/1903.10145

        :param total_steps: The total number of times annealing will be applied
        :param ratio: What proportion of eahch cycle has beta maxed out, defaults to 0.5
        :param cycles: The number of cycles which occur, defaults to 4
        """
        self.total_epochs = int(total_epochs)
        self.ratio = ratio
        self.cycles = cycles

    @property
    def is_done(self):
        return self.epoch_counter >= self.total_epochs

    def __iter__(self):
        for epoch in range(self.total_epochs):
            yield self._beta_value(self.ratio, self.cycles, epoch, self.total_epochs)

    @staticmethod
    def _beta_value(ratio: float, cycles: int, epoch: int, total_epochs: int) -> float:
        # The proportion of the schedule that has elapsed
        schedule_prop = epoch / total_epochs
        # The proportion of the current cycle that has elapsed
        cycle_prop = schedule_prop * cycles - int(schedule_prop * cycles)
        if cycle_prop > ratio:
            return 1.0
        else:
            return cycle_prop / ratio


class DeepVAELoss(LossObjective, SupervisedPlugin):
    name = "DeepVAE"

    def __init__(
        self,
        logger: av.logging.TensorboardLogger,
        beta_warmup: int,
        free_nat_constant_epochs: t.Optional[int] = None,
        free_nat_cooldown_epochs: t.Optional[int] = None,
    ) -> None:
        """Wrapper for ELBO loss for use with DeepVAE"""
        super().__init__(1.0)
        self.elbo = ELBO()
        self.logger = logger.writer

        # Beta Schedule
        self.beta_schedule = DeterministicWarmup(n=int(beta_warmup))
        self.beta_iterator: t.Optional[DeterministicWarmup] = None
        self.beta: float = 0.0

        # Fee Nat Schedule
        if free_nat_constant_epochs is not None or free_nat_cooldown_epochs is not None:
            assert (
                free_nat_constant_epochs is not None
                and free_nat_cooldown_epochs is not None
            ), (
                "Both `free_nat_constant_epochs` and `free_nat_cooldown_epochs` "
                + "must be specified"
            )
            self.free_nat_schedule = FreeNatsCooldown(
                constant_epochs=int(free_nat_constant_epochs),
                cooldown_epochs=int(free_nat_cooldown_epochs),
                start_val=2.0,
            )
            self.free_nat_iterator: t.Optional[FreeNatsCooldown] = None
            self.free_nats: float = 0.0
        else:
            self.free_nat_schedule = itertools.cycle([0.0])
            self.free_nat_iterator: t.Optional[t.Iterable] = None
            self.free_nats: float = 0.0

        self.bpd = Average()
        self.elbo_metric = Average()
        self.loss_metric = Average()
        self.likelihood = Average()
        self.kls: t.List[Average] = []

        self.before_training_exp(None)
        self.before_training_epoch(None)

    def before_training_exp(self, strategy: Strategy, *_, **kwargs):
        # Reset the warmup and free nats
        self.beta_iterator = copy.deepcopy(self.beta_schedule)
        self.free_nat_iterator = copy.deepcopy(self.free_nat_schedule)

    def before_training_epoch(self, strategy, *args, **kwargs):
        # Increment the simulated annealing
        self.beta = next(self.beta_iterator)
        self.free_nats = next(self.free_nat_iterator)
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
        self.logger.add_scalar(
            "Train.hyperparameters/beta", self.beta, strategy.clock.train_iterations
        )
        self.logger.add_scalar(
            "Train.hyperparameters/free_nats",
            self.free_nats,
            strategy.clock.train_iterations,
        )
        self.logger.add_scalar(
            "Train.likelihoods/log p(x)",
            self.elbo_metric.get(),
            strategy.clock.train_iterations,
        )
        self.logger.add_scalar(
            "Train.likelihoods/loss",
            self.loss_metric.get(),
            strategy.clock.train_iterations,
        )
        self.logger.add_scalar(
            "Train.likelihoods/log p(x|z)",
            self.likelihood.get(),
            strategy.clock.train_iterations,
        )
        self.logger.add_scalar(
            "Train.likelihoods/bpd", self.bpd.get(), strategy.clock.train_iterations
        )
        for i, kl in enumerate(self.kls):
            self.logger.add_scalar(
                f"Train.divergences/kl_z{i}", kl.get(), strategy.clock.train_iterations
            )

    def after_eval_exp(self, strategy: Strategy, *args, **kwargs):
        self.logger.add_scalar(
            f"BitsPerDim/task={self.t}", self.bpd.get(), strategy.clock.total_iterations
        )
        self.t += 1

    def update(self, out: ForwardOutput, target: Tensor = None):
        # Calculate the loss and store the likelihood and kl_divergences
        assert out.likelihood is not None, "Expected likelihood to be provided"
        assert out.kl_divergences is not None, "Expected kl_divergences to be provided"

        loss, elbo, likelihood, _ = self.elbo.forward(
            out.likelihood,
            out.kl_divergences,
            free_nats=self.free_nats,
            beta=self.beta,
            sample_reduction=torch.mean,
            batch_reduction=None,
        )
        self.loss = loss.mean()

        bpd: Tensor = -elbo.mean() / np.log(2.0) / np.prod(out.x.shape[1:])
        self.bpd.update(bpd.item())
        self.elbo_metric.update(-elbo.mean().item())
        self.loss_metric.update(loss.mean().item())
        self.likelihood.update(-likelihood.mean().item())

        for i, kl in enumerate(out.kl_divergences):
            if len(self.kls) <= i:
                self.kls.append(Average())
            self.kls[i].update(kl.mean().item())
