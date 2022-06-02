import typing as t
from .trait import Decoder, Encoder
from .wrn import WideResNetDecoder, WideResNetEncoder
from .vanilla_cnn import VanillaCNNDecoder, VanillaCNNEncoder


def vanilla_cnn(
        in_channels: int,
        latent_dims: int = 128,
        base_channels: int = 128) -> t.Tuple[Encoder, Decoder]:
    """Create a vanilla CNNs for encoding and decoding"""
    decoder = VanillaCNNDecoder(in_channels, latent_dims, base_channels)
    encoder = VanillaCNNEncoder(in_channels, latent_dims, base_channels)
    return encoder, decoder


def residual_network(
        in_channels: int,
        latent_dims: int = 128,
        base_channels: int = 128) -> t.Tuple[Encoder, Decoder]:
    return NotImplemented


def wide_residual_network(
        in_channels: int,
        latent_dims: int = 128,
        widening_factor: int = 1,
        depth: int = 6) -> t.Tuple[Encoder, Decoder]:
    """Create a wide residual network for encoding and decoding"""
    encoder = WideResNetEncoder(
        in_channels, latent_dims, widening_factor, depth)
    decoder = WideResNetDecoder(
        in_channels, latent_dims, widening_factor, depth)
    return encoder, decoder
