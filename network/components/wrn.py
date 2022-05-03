"""
Taken from https://github.com/MrtnMndt/OpenVAE_ContinualLearning/blob/master/lib/Models/architectures.py
"""
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch import nn, tensor
from torch import Tensor
from network.components.classifier import ClassifyHead, PN_ClassifyHead
from network.components.sampler import PN_VAE_Sampler, VAE_Sampler
from network.deep_generative import DAE, DVAE, PN_DVAE_InferTask
import network.module.packnet as pn
from network.trait import Decoder, Encoder

class WRNBasicBlock(nn.Module):
    """
    Convolutional or transposed convolutional block consisting of multiple 3x3 convolutions with short-cuts,
    ReLU activation functions and batch normalization.
    """
    def __init__(self, in_planes, out_planes, stride, batchnorm=1e-5, is_transposed=False):
        super(WRNBasicBlock, self).__init__()

        if is_transposed:
            self.layer1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                             output_padding=int(stride > 1), bias=False)
        else:
            self.layer1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.useShortcut = ((in_planes == out_planes) and (stride == 1))
        if not self.useShortcut:
            if is_transposed:
                self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                                   output_padding=int(1 and stride == 2), bias=False)
            else:
                self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.useShortcut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.layer1(out if self.useShortcut else x)))
        out = self.conv2(out)

        return torch.add(x if self.useShortcut else self.shortcut(x), out)


class WRNNetworkBlock(nn.Module):
    """
    Convolutional or transposed convolutional block
    """
    def __init__(self, nb_layers, in_planes, out_planes, block_type, batchnorm=1e-5, stride=1,
                 is_transposed=False):
        super(WRNNetworkBlock, self).__init__()

        if is_transposed:
            self.block = nn.Sequential(OrderedDict([
                ('convT_block' + str(layer + 1), block_type(layer == 0 and in_planes or out_planes, out_planes,
                                                             layer == 0 and stride or 1, batchnorm=batchnorm,
                                                             is_transposed=(layer == 0)))
                for layer in range(nb_layers)
            ]))
        else:
            self.block = nn.Sequential(OrderedDict([
                ('conv_block' + str(layer + 1), block_type((layer == 0 and in_planes) or out_planes, out_planes,
                                                           (layer == 0 and stride) or 1, batchnorm=batchnorm))
                for layer in range(nb_layers)
            ]))

    def forward(self, x):
        x = self.block(x)
        return x



class WRN_Encoder(Encoder, nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 widen_factor: int,
                 embedding_size: int):

        super().__init__()
        self.embedding_size = embedding_size
        self.batch_norm = 1e-5
        self.nChannels = [embedding_size, 16 * widen_factor, 32 * widen_factor,
                    64 * widen_factor, 64 * widen_factor, 64 * widen_factor,
                    64 * widen_factor]

        self.num_block_layers = 2

        self.encoder = nn.Sequential(OrderedDict([
                ('encoder_conv1',
                 nn.Conv2d(num_input_channels, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
                ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm)),
                ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_bn1', nn.BatchNorm2d(self.nChannels[3], eps=self.batch_norm)),
                ('encoder_act1', nn.ReLU(inplace=True))
            ]))

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def bottleneck_width(self) -> int:
        return self.embedding_size

    def encode(self, x: Tensor) -> Tensor:
        return self.forward(x)

class WRN_Decoder(Decoder, nn.Module):
    def __init__(
        self, 
        num_input_channels: int, 
        widen_factor: int, 
        embedding_size: int,
        latent_size: int):

        super().__init__()

        self.embedding_size = embedding_size
        self.batch_norm = 1e-5
        self.nChannels = [embedding_size, 16 * widen_factor, 32 * widen_factor,
                    64 * widen_factor, 64 * widen_factor, 64 * widen_factor,
                    64 * widen_factor]

        self.num_block_layers = 2


        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                                WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
            ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
            ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                                WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
            ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
            ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                                WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
            ('decoder_bn1', nn.BatchNorm2d(self.nChannels[0], eps=self.batch_norm)),
            ('decoder_act1', nn.ReLU(inplace=True)),
            ('decoder_conv1', nn.Conv2d(self.nChannels[0], num_input_channels, kernel_size=3, stride=1, padding=1,
                                        bias=False))
        ]))



        self.latent_decoder = nn.Linear(latent_size, 10*64*8*8, bias=False)


    def forward(self, z: Tensor):
        
        z = self.latent_decoder(z)
        z = z.view(z.size(0), 10*64, 8, 8)
        # print(z[0])
        x_hat = self.decoder(z)
        assert x_hat.isnan().count_nonzero() == 0, "HERE A"


        x_hat = torch.sigmoid(x_hat)
        return x_hat

    def decode(self, z: Tensor) -> Tensor:
        return self.forward(z)
    
    @property
    def bottleneck_width(self) -> int:
        return self.embedding_size



def construct_wrn_vae():

    return DVAE(
        WRN_Encoder(3, 10, 48),
        VAE_Sampler(10*64*8*8, 60),
        WRN_Decoder(3, 10, 48, 60),
        ClassifyHead(60, 10)

    )
    # return PN_DVAE_InferTask(
    #     WRN_Encoder(3, 5, 48),
    #     PN_VAE_Sampler(5*64*8*8, 60),
    #     WRN_Decoder(3, 5, 48, 60),
    #     PN_ClassifyHead(60, 5)

    # )




