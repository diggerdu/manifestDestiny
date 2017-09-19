# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.signal
from collections import OrderedDict


class AuFCN(nn.Module):
    def __init__(self, n_fft, hop):
        super(AuFCN, self).__init__()
        self.stft_model = stft(n_fft, hop)

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
        # pow_ inplace op
        noisy_power = (noisy_real.pow(2.) + noisy_imag.pow(2.)).pow(0.5)
        #	noisy_angle = torch.atan2(noisy_imag, noisy_real)
        estimated_clean_frame = self.fcn(noisy_power).squeeze()

        return estimated_clean_frame


class stft(nn.Module):
    def __init__(self, nfft=1024, hop_length=512, window="hanning"):
        super(stft, self).__init__()
        assert nfft % 2 == 0

        self.hop_length = hop_length
        self.n_freq = n_freq = nfft // 2 + 1

        self.real_kernels, self.imag_kernels = _get_stft_kernels(nfft, window)

    def forward(self, sample):
        sample = sample.unsqueeze(1)
        sample = sample.unsqueeze(1)

        magn = F.conv2d(sample, self.real_kernels, stride=self.hop_length)
        phase = F.conv2d(sample, self.imag_kernels, stride=self.hop_length)
        print(self.real_kernels.data.shape)
        print('magn shape ', magn.data.shape)
        magn = magn.permute(0, 3, 1, 2)
        phase = phase.permute(0, 3, 1, 2)

        # complex conjugate
        phase = -1 * phase[:, :, 1:, :]
        ac = magn[:, :, 0, :]
        magn = magn[:, :, 1:, :]
        print('after magn shape', magn.data.shape)
        return magn, phase, ac


def _get_stft_kernels(nfft, window):
    nfft = int(nfft)
    assert nfft % 2 == 0

    def kernel_fn(freq, time):
        return np.exp(-1j * (2 * np.pi * time * freq) / float(nfft))

    kernels = np.fromfunction(
        kernel_fn, (nfft // 2 + 1, nfft), dtype=np.float64)

    if window == "hanning":
        win_cof = scipy.signal.get_window("hanning", nfft)[np.newaxis, :]
    else:
        win_cof = np.ones((1, nfft), dtype=np.float64)

    kernels = kernels[:, np.newaxis, np.newaxis, :] * win_cof

    real_kernels = nn.Parameter(torch.from_numpy(np.real(kernels)).float())
    imag_kernels = nn.Parameter(torch.from_numpy(np.imag(kernels)).float())

    return real_kernels, imag_kernels


sample = Variable(torch.FloatTensor(3, 1664), requires_grad=True)
model = AuFCN(256, 128)
result = model(sample)
print(result.data.shape, 'result_shape', type(result.data))
criterion = nn.MSELoss()
ref = Variable(torch.FloatTensor(3, 128))
loss = criterion(result, ref)
print('loss', loss.data[0])
B = Variable(torch.Tensor(3, 256)).cuda()
Bstft = stft(nfft=256, hop_length=128)
B_then = Bstft(B)
