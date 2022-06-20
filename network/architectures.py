from dataclasses import dataclass
import typing as t
from .trait import Decoder, Encoder
from .wrn import WideResNetDecoder, WideResNetEncoder
from .vanilla_cnn import VanillaCNNDecoder, VanillaCNNEncoder


@dataclass
class AEArchitecture():
    encoder: Encoder
    decoder: Decoder
    latent_dims: int

def vanilla_cnn(
        in_channels: int,
        latent_dims: int,
        base_channels: int) -> AEArchitecture:
    """Create a vanilla CNNs for encoding and decoding"""
    decoder = VanillaCNNDecoder(in_channels, base_channels, latent_dims)
    encoder = VanillaCNNEncoder(in_channels, base_channels, latent_dims)
    return AEArchitecture(encoder, decoder, latent_dims)


def residual_network(
        in_channels: int,
        latent_dims: int = 128,
        base_channels: int = 128) -> AEArchitecture:
    return NotImplemented


def wide_residual_network(
        in_channels: int,
        latent_dims: int = 128,
        widening_factor: int = 1,
        depth: int = 8) -> AEArchitecture:
    """Create a wide residual network for encoding and decoding"""
    encoder = WideResNetEncoder(
        in_channels, latent_dims, widening_factor, depth//2)
    decoder = WideResNetDecoder(
        in_channels, latent_dims, widening_factor, depth//2)
    return AEArchitecture(encoder, decoder, latent_dims)