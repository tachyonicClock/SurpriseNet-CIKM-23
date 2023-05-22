import typing as t
from network.deep_vae import FashionMNISTDeepVAE
from config.config import ExpConfig
from network.mlp import (
    ClassifierHead,
    MLPDecoder,
    MLPEncoder,
    MLPDecoder,
    MLPEncoder,
    VAEBottleneck,
)

from network.resnet import ResNet18Decoder, ResNet18Encoder
from .trait import AutoEncoder, Decoder, Encoder, VariationalAutoEncoder
from .vanilla_cnn import VanillaCNNDecoder, VanillaCNNEncoder

_NETWORK_ARCHITECTURES = {
    "vanilla_cnn": (VanillaCNNEncoder, VanillaCNNDecoder),
    "residual": (ResNet18Encoder, ResNet18Decoder),
    "mlp": (MLPEncoder, MLPDecoder),
}


def construct_network(cfg: ExpConfig):
    """
    Construct an auto encoder based on the configuration
    """
    latent_dims = int(cfg.latent_dims)
    if cfg.network_style == "DeepVAE_FMNIST":
        return FashionMNISTDeepVAE(cfg.n_classes, latent_dims, **cfg.network_cfg)

    encoder_constructor = _NETWORK_ARCHITECTURES[cfg.network_style][0]
    decoder_constructor = _NETWORK_ARCHITECTURES[cfg.network_style][1]

    if cfg.architecture == "AE":
        encoder: Encoder = encoder_constructor(
            latent_dims, cfg.input_shape, **cfg.network_cfg
        )
        decoder: Decoder = decoder_constructor(
            latent_dims, cfg.input_shape, **cfg.network_cfg
        )
        classifier = ClassifierHead(latent_dims, cfg.n_classes)
        return AutoEncoder(encoder, decoder, classifier)
    elif cfg.architecture == "VAE":
        encoder: Encoder = encoder_constructor(
            latent_dims * 2, cfg.input_shape, **cfg.network_cfg
        )
        decoder: Decoder = decoder_constructor(
            latent_dims, cfg.input_shape, **cfg.network_cfg
        )
        classifier = ClassifierHead(latent_dims, cfg.n_classes)
        bottleneck = VAEBottleneck(latent_dims * 2, latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, classifier)
    assert False, "Unknown architecture"
