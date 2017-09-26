import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import time_frequence as tf


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.gan_loss = opt.gan_loss
        self.isTrain = opt.isTrain

        # define tensors self.Tensor has been reloaded
        self.input_A = self.Tensor(opt.batchSize, opt.len)
        self.input_B = self.Tensor(opt.batchSize, opt.nfft)
        if self.gpu_ids:
            self.input_A.cuda(device=self.gpu_ids[0])
            self.input_B.cuda(device=self.gpu_ids[0])

        # load/define networks
        self.netG = networks.define_G(opt.nfft, opt.hop, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                              opt.which_model_netD,
            #                              opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            # if self.isTrain:
            #    self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterion = torch.nn.MSELoss()

            # initialize optimizers

            self.TrainableParam = list()
            param = self.netG.named_parameters()
            IgnoredParam = [id(P) for name, P in param if 'stft' in name]

            ########################

            #self.optimizer_G = torch.optim.Adam(self.TrainableParam,
            #                                   lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(
                filter(lambda P: id(P) not in IgnoredParam,
                       self.netG.parameters()),
                lr=opt.lr,
                betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
            #lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            # networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']
        ########### debug #############

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(
            torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # self.loss_G = self.criterionL1(self.fake_B, Variable(torch.cuda.FloatTensor(self.fake_B.size()).zero_()))

        self.loss_G = self.criterion(self.fake_B, self.real_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if self.gan_loss:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.gan_loss:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0])])
        else:
            #print("#############clean sample mean#########")
            #sample_data = self.input_B.cpu().numpy()

            #print("max value", np.max(sample_data))
            #print("mean value", np.mean(np.abs(sample_data)))
            return OrderedDict([('G_LOSS', self.loss_G.data[0])])

    def get_current_visuals(self):
        real_A = self.real_A.data.cpu().numpy()
        fake_B = self.fake_B.data.cpu().numpy()
        real_B = self.real_B.data.cpu().numpy()
        clean = self.clean.cpu().numpy()
        noise = self.noise.cpu().numpy()
        return OrderedDict([
            ('est_ratio', fake_B),
            ('clean', clean),
            ('ratio', real_B),
            ('noise', noise),
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.gan_loss:
            self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        # lrd = self.opt.lr / self.opt.niter_decay
        # lr = self.old_lr - lrd
        lr = self.old_lr * 0.6
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
