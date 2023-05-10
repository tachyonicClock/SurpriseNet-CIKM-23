from avalanche.core import SupervisedPlugin
from experiment.strategy import ForwardOutput
from network.hvae.oodd.losses import ELBO
from .trait import AutoEncoder, Encoder, Decoder, VariationalAutoEncoder, Samplable, MultiOutputNetwork
from network.hvae.oodd.layers.stages import VaeStage, LvaeStage
from network.hvae.oodd.models.dvae import DeepVAE
from network.hvae.oodd.variational import DeterministicWarmup, FreeNatsCooldown
from torch import nn, Tensor
import torch
from experiment.loss import LossObjective
import avalanche as av


class FashionMNISTDeepVAE(Encoder, Decoder, Samplable, MultiOutputNetwork):

    def __init__(self) -> None:
        super().__init__()

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
        self.dummy = nn.Parameter(torch.zeros(1))
        self.deep_vae = DeepVAE(
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

    def forward(self, x):
        return self.deep_vae.forward(x, n_posterior_samples=1)

    def classify(self, x: Tensor) -> Tensor:
        raise NotImplementedError("DeepVAE does not support classification")

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError("DeepVAE does not support encoding")

    def decode(self, posteriors: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError("DeepVAE does not support decoding")

    def sample(self, n: int) -> Tensor:
        likelihood_data, _ = self.deep_vae.sample_from_prior(n)
        samples: Tensor = likelihood_data.samples.view(n, *self.deep_vae.input_shape)
        return samples

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def multi_forward(self, x: Tensor) -> ForwardOutput:
        likelihood_data, stage_data = self.deep_vae.forward(
            x, n_posterior_samples=1)
        kl_divergences = [
            sd.loss.kl_elementwise for sd in stage_data if sd.loss.kl_elementwise is not None
        ]

        forward_output = ForwardOutput()
        forward_output.x_hat = likelihood_data.samples
        forward_output.likelihood = likelihood_data.likelihood
        forward_output.kl_divergences = kl_divergences

        return forward_output


class DeepVAELoss(LossObjective, SupervisedPlugin):
    name = "DeepVAE"

    def __init__(self, 
            weighting: float,
            free_nat_start_value: float,
            free_nats_epochs: float,
            warmup_epochs: int,
            logger:  av.logging.TensorboardLogger
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
        self.warmup_epochs = warmup_epochs
        self.logger = logger.writer
        self.before_training_exp(None)
        self.before_training_epoch(None)

    def before_training_exp(self, strategy: 'Template', *args, **kwargs):
        print("DeepVAE Loss: Initializing warmup and free nats")
        # Reset the warmup and free nats
        self.deterministic_warmup = DeterministicWarmup(n=self.warmup_epochs)
        self.free_nats_cool_down = FreeNatsCooldown(
            constant_epochs=self.free_nats_epochs // 2,
            cooldown_epochs=self.free_nats_epochs // 2,
            start_val=self.free_nat_start_value,
            end_val=0,
        )

    def before_training_epoch(self, strategy: 'Template', *args, **kwargs):
        # Increment the simulated annealing
        self.beta = next(self.deterministic_warmup)
        self.free_nats = next(self.free_nats_cool_down)
        self.logger.add_scalar("ELBO/beta", self.beta)
        self.logger.add_scalar("ELBO/free_nats", self.free_nats)

    def update(self, out: ForwardOutput, target: Tensor = None):
        # Calculate the loss and store the likelihood and kl_divergences
        assert out.likelihood is not None, "Expected likelihood to be provided"
        assert out.kl_divergences is not None, "Expected kl_divergences to be provided"

        loss, elbo, likelihood, kl_divergences = self.elbo.forward(
            out.likelihood,
            out.kl_divergences,
            free_nats=self.free_nats,
            beta=self.beta,
            sample_reduction=torch.mean,
            batch_reduction=None
        )
        self.loss = loss.mean()
