import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules import Module
import numpy as np
from collections import OrderedDict

from . import time_frequence as tf

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find(
            'InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm)
    return norm_layer
    # return None


def define_G(n_fft, hop, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = AuFCNWrapper(n_fft, hop, gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc,
             ndf,
             which_model_netD,
             n_layers_D=3,
             norm='batch',
             use_sigmoid=False,
             gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers=3,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
            gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class AuFCNWrapper(nn.Module):
    def __init__(self, n_fft, hop, gpu_ids):
        super(AuFCNWrapper, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = AuFCN(n_fft, hop)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data,
                                       torch.cuda.FloatTensor) and False:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            print("network G output", output.size())
            return output
        else:
            return self.model(input)


# TODO robust
# TODO requires gradient
# TODO assert AC == 0
class AuFCN(nn.Module):
    def __init__(self, n_fft, hop):
        super(AuFCN, self).__init__()
        self.stft_model = tf.stft(n_fft, hop)
        self.istft_model = tf.istft(n_fft, hop)

        fcn = None

        fcn = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(
                12, 52, kernel_size=(5, 1), padding=1)), ('relu1', nn.ReLU(
                    True)), ('maxpool', nn.MaxPool2d(3)), ('conv2', nn.Conv2d(
                        52, 78, kernel_size=(5, 1), padding=0)), (
                            'relu2', nn.ReLU(True)), ('fc1', nn.Conv2d(
                                78, 1024, kernel_size=(38, 1))), (
                                    'relu_after_linear1', nn.ReLU(True)),
                         ('fc2', nn.Conv2d(1024, 1024, kernel_size=(1, 1))), (
                             'relu_after_linear2', nn.ReLU(True)),
                         ('fc3', nn.Conv2d(1024, 128, kernel_size=(1, 1)))]))
        self.fcn = fcn

    def forward(self, sample):
        noisy_real, noisy_imag, ac = self.stft_model(sample)
        noisy_angle = torch.atan2(noisy_imag, noisy_real)
        noisy_power = (noisy_real.pow(2.) + noisy_imag.pow(2.)).pow(0.5)

        IRM = self.fcn(noisy_power).permute(0, 2, 1, 3)

        clean_power = IRM * noisy_power[:, 6:7, :, :]
        clean_real = clean_power * torch.cos(noisy_angle[:, 6:7, :, :])
        clean_imag = clean_power * torch.sin(noisy_angle[:, 6:7, :, :])
        estimated_clean_frame = self.istft_model(clean_real, clean_imag,
                                                 ac[:, 6:7, :])
        '''
		#############DEBUG#############
		power_debug = clean_power.data.cpu().numpy()
		count = np.sum(np.isnan(power_debug))
		print("nan count", count)
		###############################
		'''

        # clean_sample = self.istft_model(noisy_magn, noisy_phase, ac)

        #sample_debug = clean_sample.data.cpu().numpy()
        #print("max G output", np.max(sample_debug))
        #print("min G output", np.min(sample_debug))
        '''
		print("#########noisy angle debug###########")
		magn_debug = clean_magn.data.cpu().numpy()
		phase_debug = clean_phase.data.cpu().numpy()
		angle_debug = noisy_angle.data.cpu().numpy()
		debug = (noisy_magn / noisy_power).data.cpu().numpy()
		sample_debug = clean_sample.data.cpu().numpy()
		# print(np.max(debug))
		# print(np.min(debug))
		# print(np.max(magn_debug))
		# print(np.max(angle_debug))
		# print(angle_debug[np.argmax(np.isnan(angle_debug))])
		print(np.sum(np.isnan(magn_debug)))
		print(np.sum(np.isnan(phase_debug)))
		print(np.sum(np.isnan(sample_debug)))
		print(np.max(sample_debug))
		print(np.min(sample_debug))

		import sys
		sys.exit(1)
		'''

        return estimated_clean_frame


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            norm_layer(ngf, affine=True),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                # norm_layer(ngf * mult * 2, affine=True),
                nn.ReLU(True)
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    'zero',
                    norm_layer=norm_layer,
                    use_dropout=use_dropout)
            ]

        # TODO
        # model += [nn.ConvTranspose2d(512, 256, kernel_size=(3,3), stride=(1, 1), padding=(1,1), output_padding=(1,1))]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [
                torch.nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=1,
                    padding=1),
                # norm_layer(int(ngf * mult / 2), affine=True),
                nn.ReLU(True)
            ]

        # model += [nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class WideResnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False,
                 n_blocks=6):
        assert (n_blocks >= 0)
        super(WideResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=2, padding=(0, 1)),
            norm_layer(ngf, affine=True),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(ngf, ngf * 2, kernel_size=2, padding=(1, 0)),
            norm_layer(ngf * 2, affine=True),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**(i + 1)
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                # norm_layer(ngf * mult * 2, affine=True),
                nn.ReLU(True)
            ]

        mult = 2**(n_downsampling + 1)
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    'zero',
                    norm_layer=norm_layer,
                    use_dropout=use_dropout)
            ]

        # TODO
        # model += [nn.ConvTranspose2d(512, 256, kernel_size=(3,3), stride=(1, 1), padding=(1,1), output_padding=(1,1))]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling + 1 - i)
            model += [
                torch.nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=1,
                    padding=1),
                # norm_layer(int(ngf * mult / 2), affine=True),
                nn.ReLU(True)
            ]

        # model += [nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)]
        model += [nn.Conv2d(ngf * 2, output_nc, kernel_size=3, padding=1)]
        model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                                use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            # norm_layer(dim, affine=True),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            # norm_layer(dim, affine=True)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self,
                 input_nc,
                 output_nc,
                 num_downs,
                 ngf=64,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetGenerator, self).__init__()

        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_nc,
                 inner_nc,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(
            outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, Tanh_rescale()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self,
                 input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw),
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw),
            # TODO: useInstanceNorm
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data,
                                            torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Tanh_rescale(Module):
    def forward(self, input):
        return torch.div(
            torch.add(torch.tanh(torch.mul(input, 2.0)), 1.0), 2.0)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
