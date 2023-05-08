import typing as t
from network.deep_vae import FashionMNISTDeepVAE
from config.config import ExpConfig
from network.mlp import ClassifierHead, MLPDecoder, MLPEncoder, MLPDecoder, MLPEncoder, VAEBottleneck

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
    if cfg.network_style == "DeepVAE_FMNIST":
        return FashionMNISTDeepVAE()

    encoder_constructor = _NETWORK_ARCHITECTURES[cfg.network_style][0]
    decoder_constructor = _NETWORK_ARCHITECTURES[cfg.network_style][1]

    if cfg.architecture == "AE":
        encoder: Encoder = encoder_constructor(cfg.latent_dims, cfg.input_shape, **cfg.network_cfg)
        decoder: Decoder = decoder_constructor(cfg.latent_dims, cfg.input_shape, **cfg.network_cfg)
        classifier = ClassifierHead(cfg.latent_dims, cfg.n_classes)
        return AutoEncoder(encoder, decoder, classifier)
    elif cfg.architecture == "VAE":
        encoder: Encoder = encoder_constructor(cfg.latent_dims*2, cfg.input_shape, **cfg.network_cfg)
        decoder: Decoder = decoder_constructor(cfg.latent_dims, cfg.input_shape, **cfg.network_cfg)
        classifier = ClassifierHead(cfg.latent_dims, cfg.n_classes)
        bottleneck = VAEBottleneck(cfg.latent_dims*2, cfg.latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, classifier)
    assert False, "Unknown architecture"
