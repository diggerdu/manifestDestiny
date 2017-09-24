import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.audio_process import recovery_phase

from pdb import set_trace as st
from util import html


# audio process
import numpy as np
import librosa
import soundfile as sf


def loadAudio(self, path, SR):
    data, sr = sf.read(path)
    try:
        assert sr = SR
    except AssertionError:
        data = librosa.resample(data, sr, opt.SR)
    return data - np.mean(data)

def eval(cleanPath, noisePath, snr):
    clean = loadAudio(cleanPath)
    noise = loadAudio(noisePath)
    noise = np.tile(noise, clean.shape[0] // noise.shape[0] + 1)[:clean.shape[0]]
    noiseAmp = np.mean(np.square(clean)) / np.power(10, snr / 10.0)
    scale = np.sqrt(noiseAmp / np.mean(np.square(noise)))
    mix = clean + scale * noise








opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

model = create_model(opt)


