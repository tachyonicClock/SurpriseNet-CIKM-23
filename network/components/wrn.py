from collections import OrderedDict
from functional.task_inference import infer_task
import network.module.packnet as pn
import torch
import torch.nn as nn
from experiment.strategy import ForwardOutput

from network.trait import AutoEncoder, Classifier, ConditionedSample, InferTask, PackNetComposite, Samplable

def wrn_bn(features, eps):
    return nn.BatchNorm2d(features, eps=eps)

def get_feat_size(block, spatial_size, ncolors=3):
    """
    Function to infer spatial dimensionality in intermediate stages of a model after execution of the specified block.

    Parameters:
        block (torch.nn.Module): Some part of the model, e.g. the encoder to determine dimensionality before flattening.
        spatial_size (int): Quadratic input's spatial dimensionality.
        ncolors (int): Number of dataset input channels/colors.
    """

    x = torch.randn(2, ncolors, spatial_size, spatial_size)
    out = block(x)
    num_feat = out.size(1)
    spatial_dim_x = out.size(2)
    spatial_dim_y = out.size(3)

    return num_feat, spatial_dim_x, spatial_dim_y

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
        self.bn1 = wrn_bn(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = wrn_bn(out_planes, eps=batchnorm)
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


class WRN(AutoEncoder, Classifier, Samplable, nn.Module):
    """
    Flexibly sized Wide Residual Network (WRN). Extended to the variational setting and to our unified model.
    """
    def __init__(self, 
        num_classes: int, 
        num_colors: int,
        widen_factor: int = 10,
        batch_norm_eps: float = 1e-05,
        embedding_size: int = 48,
        var_latent_dim: int = 60,
        depth: int = 14, 
        patch_size: int = 32,
        batch_size: int = 32,
        out_channels: int = 3,
        double_wrn_blocks: bool = False):
        super().__init__()

        self.widen_factor = widen_factor
        self.depth = depth
        self.batch_norm = batch_norm_eps
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_colors = num_colors
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.double_blocks = double_wrn_blocks

        self.num_samples = 1
        self.latent_dim = var_latent_dim

        self.nChannels = [embedding_size, 16 * self.widen_factor, 32 * self.widen_factor,
                          64 * self.widen_factor, 64 * self.widen_factor, 64 * self.widen_factor,
                          64 * self.widen_factor]

        if self.double_blocks:
            assert ((self.depth - 2) % 12 == 0)
            self.num_block_layers = int((self.depth - 2) / 12)
            self.encoder = nn.Sequential(OrderedDict([
                ('encoder_conv1',
                 nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
                ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block4', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[4],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block5', WRNNetworkBlock(self.num_block_layers, self.nChannels[4], self.nChannels[5],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block6', WRNNetworkBlock(self.num_block_layers, self.nChannels[5], self.nChannels[6],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_bn1', wrn_bn(self.nChannels[6], eps=self.batch_norm)),
                ('encoder_act1', nn.ReLU(inplace=True))
            ]))
        else:
            assert ((self.depth - 2) % 6 == 0)
            self.num_block_layers = int((self.depth - 2) / 6)

            self.encoder = nn.Sequential(OrderedDict([
                ('encoder_conv1',
                 nn.Conv2d(num_colors, self.nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)),
                ('encoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[0], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm)),
                ('encoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=2)),
                ('encoder_bn1', wrn_bn(self.nChannels[3], eps=self.batch_norm)),
                ('encoder_act1', nn.ReLU(inplace=True))
            ]))

        self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y = get_feat_size(self.encoder, self.patch_size,
                                                                                          self.num_colors)

        self.latent_mu = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_x * self.enc_channels,
                                   self.latent_dim, bias=False)
        self.latent_std = nn.Linear(self.enc_spatial_dim_x * self.enc_spatial_dim_y * self.enc_channels,
                                    self.latent_dim, bias=False)

        self.classifier = nn.Sequential(nn.Linear(self.latent_dim, num_classes, bias=False))

        self.latent_decoder = nn.Linear(self.latent_dim, self.enc_spatial_dim_x * self.enc_spatial_dim_y *
                                        self.enc_channels, bias=False)

        if self.double_blocks:
            self.decoder: torch.Tensor = nn.Sequential(OrderedDict([
                ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[6], self.nChannels[5],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[5], self.nChannels[4],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[4], self.nChannels[3],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample3', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block4', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample4', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block5', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample5', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block6', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_bn1', wrn_bn(self.nChannels[0], eps=self.batch_norm)),
                ('decoder_act1', nn.ReLU(inplace=True)),
                ('decoder_upsample6', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_conv1', nn.Conv2d(self.nChannels[0], self.out_channels, kernel_size=3, stride=1, padding=1,
                                            bias=False))
            ]))
        else:
            self.decoder: torch.Tensor = nn.Sequential(OrderedDict([
                ('decoder_block1', WRNNetworkBlock(self.num_block_layers, self.nChannels[3], self.nChannels[2],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample1', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block2', WRNNetworkBlock(self.num_block_layers, self.nChannels[2], self.nChannels[1],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_upsample2', nn.Upsample(scale_factor=2, mode='nearest')),
                ('decoder_block3', WRNNetworkBlock(self.num_block_layers, self.nChannels[1], self.nChannels[0],
                                                   WRNBasicBlock, batchnorm=self.batch_norm, stride=1)),
                ('decoder_bn1', wrn_bn(self.nChannels[0], eps=self.batch_norm)),
                ('decoder_act1', nn.ReLU(inplace=True)),
                ('decoder_conv1', nn.Conv2d(self.nChannels[0], self.out_channels, kernel_size=3, stride=1, padding=1,
                                            bias=False))
            ]))
        self.dummy_param = nn.Parameter(torch.empty(0))


    @property
    def device(self):
        return self.dummy_param.device

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean = self.latent_mu(x)
        z_std = self.latent_std(x)
        return z_mean, z_std

    def reparameterize(self, mu, std):
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add(mu)

    def decode(self, z):
        z = self.latent_decoder(z)
        z = z.view(z.size(0), self.enc_channels, self.enc_spatial_dim_x, self.enc_spatial_dim_y)
        x_hat = self.decoder(z)
        x_hat = torch.sigmoid(x_hat)
        return x_hat

    def generate(self):
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = self.decode(z)
        return x

    def forward(self, x):
        
        z_mean, z_std = self.encode(x)

        output_samples = torch.zeros(self.num_samples, x.size(0), self.out_channels, self.patch_size,
                                     self.patch_size).to(self.device)
        classification_samples = torch.zeros(self.num_samples, x.size(0), self.num_classes).to(self.device)
        for i in range(self.num_samples):
            z = self.reparameterize(z_mean, z_std)
            output_samples[i] = self.decode(z)
            classification_samples[i] = self.classifier(z)

        out = ForwardOutput()
        out.y_hat = classification_samples[0]
        out.x_hat = output_samples[0]
        out.x = x
        out.mu = z_mean
        out.log_var = z_std.square().log()

        return out

    @property
    def bottleneck_width(self):
        return self.latent_dim

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).y_hat

    def sample(self, n: int = 1) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim).to(self.device)
        x = self.decode(z)
        return x


class PN_WRN(PackNetComposite, ConditionedSample,  WRN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pn.wrap(self)

    def conditioned_sample(self, n: int = 1, given_task: int = 0) -> torch.Tensor:
        self.use_task_subset(given_task)
        sample = self.sample(n)
        self.use_top_subset()
        return sample

class PN_DVAE_WRN_InferTask(InferTask, PN_WRN):

    subnet_count: int = 0

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        if self.training:
            return super().forward(x)
        return infer_task(super(), x, self.subnet_count, 2)

    def push_pruned(self):
        super().push_pruned()
        self.subnet_count += 1
        


# class WRN(AutoEncoder, Classifier, Samplable, _WRN):

#     def forward(self, x: torch.Tensor) -> ForwardOutput:
#         out = ForwardOutput()
#         classification_samples, output_samples, z_mean, z_std = super().forward(x)
#         out.x_hat = output_samples[0]
#         out.x = x
#         out.y_hat = classification_samples[0]
#         out.mu = z_mean
#         out.log_var = z_std.square().log()
#         return out

