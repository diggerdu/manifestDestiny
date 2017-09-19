import os.path
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset
import librosa
import soundfile as sf
import numpy as np
import random


class AudioDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.DirClean = opt.PathClean
        self.DirNoise = opt.PathNoise
        self.snr = opt.snr

        self.FilesClean = make_dataset(self.DirClean)
        self.FilesNoise = make_dataset(self.DirNoise)

        self.FilesClean = sorted(self.FilesClean)
        self.FilesNoise = sorted(self.FilesNoise)
        self.SR = opt.SR
        self.hop = opt.hop
        self.nfft = self.opt.nfft
        self.B_start = 6 * self.opt.hop

    def __getitem__(self, index):
        Clean = self.FilesClean[index]
        Noise = self.FilesNoise[0]

        CleanAudio, head = self.load_audio(Clean)
        NoiseAudio, _ = self.load_audio(Noise)
        A = self.addnoise(CleanAudio, NoiseAudio)



        return {
            'A': A,
            'B': CleanAudio[self.B_start:self.B_start + self.nfft],
            'A_paths': Clean
        }

    def __len__(self):
        return len(self.FilesClean)

    def addnoise(self, clean, noise):
        assert clean.shape == noise.shape
        noiseAmp = np.mean(np.square(clean)) / np.power(10, self.snr / 10.0)
        scale = np.sqrt(noiseAmp / np.mean(np.square(noise)))
        return clean + scale * noise

    def name(self):
        return "AudioDataset"

    def load_audio(self, path):
        data, samplerate = sf.read(path)
        try:
            assert samplerate == self.SR and len(data.shape) == 1
        except AssertionError:
            data = librosa.resample(data, samplerate, self.SR)

        target_len = self.opt.len
        if data.shape[0] >= target_len:
            head = random.randint(0, data.shape[0] - target_len)
            data = data[head:head + target_len]
        if data.shape[0] < target_len:
            ExtraLen = target_len - data.shape[0]
            PrevExtraLen = np.random.randint(ExtraLen)
            PostExtraLen = ExtraLen - PrevExtraLen
            PrevExtra = np.zeros((PrevExtraLen, ))
            PostExtra = np.zeros((PostExtraLen, ))
            data = np.concatenate((PrevExtra, data, PostExtra))

        # nomarlize level
        data = data - np.mean(data)

        assert data.shape[0] == self.opt.len
        return data, head
