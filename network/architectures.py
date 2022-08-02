from dataclasses import dataclass
import typing as t
from network.mlp import MLPDecoder, MLPEncoder, UniformMLPDecoder, UniformMLPEncoder, MPLRectangularClassifierHead

from network.resnet import ResNet18Dec, ResNet18Enc
from .trait import AutoEncoder, Classifier, Decoder, Encoder, VariationalAutoEncoder
from .wrn import WideResNetDecoder, WideResNetEncoder
from .vanilla_cnn import ClassifierHead, VAEBottleneck, VanillaCNNDecoder, VanillaCNNEncoder


@dataclass
class AEArchitecture():
    encoder: Encoder
    decoder: Decoder
    head: Classifier
    latent_dims: int

def vanilla_cnn(
        n_classes: int,
        in_channels: int,
        latent_dims: int,
        base_channels: int,
        is_vae: bool = False) -> t.Union[AutoEncoder, VariationalAutoEncoder]:
    """Create a vanilla CNNs for encoding and decoding"""
    # VAE uses VAEBottleneck as the bottleneck so we need more channels here
    encoder_output_dims = latent_dims*2 if is_vae else latent_dims

    encoder = VanillaCNNEncoder(in_channels, base_channels, encoder_output_dims)
    decoder = VanillaCNNDecoder(in_channels, base_channels, latent_dims)
    head = ClassifierHead(latent_dims, n_classes)

    if is_vae:
        bottleneck = VAEBottleneck(encoder_output_dims, latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, head)
    else:
        return AutoEncoder(encoder, decoder, head)


def residual_network(
        n_classes: int,
        latent_dims: int = 128,
        image_shape: t.Tuple[int, int, int] = (3, 32, 32),
        is_vae: bool = False) -> AEArchitecture:
    encoder_output_dims = latent_dims if not is_vae else latent_dims*2
    decoder_input_dims = latent_dims

    encoder = ResNet18Enc(z_dim=encoder_output_dims, shape=image_shape)
    decoder = ResNet18Dec(z_dim=decoder_input_dims, shape=image_shape)
    head = ClassifierHead(latent_dims, n_classes)
    if is_vae:
        bottleneck = VAEBottleneck(encoder_output_dims, latent_dims)
        return VariationalAutoEncoder(encoder, bottleneck, decoder, head)
    else:
        return AutoEncoder(encoder, decoder, head)

def mlp_network(
        n_classes: int,
        in_dimensions: int,
        latent_dims: int = 128,
        vae: bool = False) -> AEArchitecture:
    """Create a MLP for encoding and decoding"""
    encoder_output_dims = latent_dims if not vae else latent_dims*2
    decoder_input_dims = latent_dims

    encoder = MLPEncoder(in_dimensions, encoder_output_dims)
    decoder = MLPDecoder(decoder_input_dims, in_dimensions)
    head = ClassifierHead(latent_dims, n_classes)
    return AEArchitecture(encoder, decoder, head, latent_dims)

def rectangular_network(
        n_classes: int,
        in_features: int,
        latent_dims: int,
        depth: int,
        width: int,
        vae: bool = False) -> AEArchitecture:
    """Create a rectangular network for encoding and decoding"""
    encoder_output_dims = latent_dims if not vae else latent_dims*2
    decoder_input_dims = latent_dims

    encoder = UniformMLPEncoder(in_features, width, depth, encoder_output_dims)
    decoder = UniformMLPDecoder(in_features, width, depth, decoder_input_dims)
    head = MPLRectangularClassifierHead(latent_dims, width, n_classes)
    return AEArchitecture(encoder, decoder, head, latent_dims)

def wide_residual_network(
        n_classes: int,
        in_channels: int,
        latent_dims: int = 128,
        widening_factor: int = 1,
        depth: int = 8,
        vae: bool = False) -> AEArchitecture:
    """Create a wide residual network for encoding and decoding"""
    encoder_output_dims = latent_dims if not vae else latent_dims*2
    decoder_input_dims = latent_dims

    encoder = WideResNetEncoder(
        encoder_output_dims, latent_dims, widening_factor, depth//2)
    decoder = WideResNetDecoder(
        decoder_input_dims, latent_dims, widening_factor, depth//2)
    head = ClassifierHead(latent_dims, n_classes)
    return AEArchitecture(encoder, decoder, head, latent_dims)