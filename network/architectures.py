import typing as t
from network.mlp import MLPDecoder, MLPEncoder, MLPDecoder, MLPEncoder

from network.resnet import ResNet18Dec, ResNet18Enc
from .trait import AutoEncoder, VariationalAutoEncoder
from .vanilla_cnn import ClassifierHead, VAEBottleneck, VanillaCNNDecoder, VanillaCNNEncoder
import torch

def construct_vanilla_cnn(
        n_classes: int,
        latent_dims: int,
        input_shape: torch.Size,
        vae: bool,
        cfg: t.Dict[str, t.Any]) -> t.Union[AutoEncoder, VariationalAutoEncoder]:
    """Create a vanilla CNNs for encoding and decoding"""
    # VAE uses VAEBottleneck as the bottleneck so we need more channels here
    encoder_output_dims = latent_dims*2 if vae else latent_dims

    if "base_channels" not in cfg:
        raise ValueError("base_channels must be provided in config")
    
    base_channels = cfg["base_channels"]
    in_channels = input_shape[0]

    encoder = VanillaCNNEncoder(in_channels, base_channels, encoder_output_dims)
    decoder = VanillaCNNDecoder(in_channels, base_channels, latent_dims)
    head = ClassifierHead(latent_dims, n_classes)

    if vae:
        bottleneck = VAEBottleneck(encoder_output_dims, latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, head)
    else:
        return AutoEncoder(encoder, decoder, head)

def construct_resnet18_cnn(
        n_classes: int,
        latent_dims: int,
        input_shape: torch.Size,
        vae: bool,
        cfg: t.Dict[str, t.Any]) -> t.Union[AutoEncoder, VariationalAutoEncoder]:
    encoder_output_dims = latent_dims if not vae else latent_dims*2
    decoder_input_dims = latent_dims

    encoder = ResNet18Enc(z_dim=encoder_output_dims, shape=input_shape)
    decoder = ResNet18Dec(z_dim=decoder_input_dims, shape=input_shape)
    head = ClassifierHead(latent_dims, n_classes)
    if vae:
        bottleneck = VAEBottleneck(encoder_output_dims, latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, head)
    else:
        return AutoEncoder(encoder, decoder, head)

def construct_mlp_network(
        n_classes: int,
        latent_dims: int,
        input_shape: torch.Size,
        vae: bool,
        cfg: t.Dict[str, t.Any]) -> t.Union[AutoEncoder, VariationalAutoEncoder]:
    """Create a MLP for encoding and decoding"""
    encoder_output_dims = latent_dims if not vae else latent_dims*2
    decoder_input_dims = latent_dims

    if "width" not in cfg:
        raise ValueError("width must be provided in config")
    width = cfg["width"]

    encoder = MLPEncoder(input_shape, width, encoder_output_dims)
    decoder = MLPDecoder(input_shape, width, decoder_input_dims)
    head = ClassifierHead(latent_dims, n_classes)
    if vae:
        bottleneck = VAEBottleneck(encoder_output_dims, latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, head)
    else:
        return AutoEncoder(encoder, decoder, head)
