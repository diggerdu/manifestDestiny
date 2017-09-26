import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.audio_process import recovery_phase
import torch

from pdb import set_trace as st
from util import html


# audio process
import numpy as np
import librosa
import soundfile as sf


def CalSNR(ref, sig):
    ref_p = np.square(ref)
    noi_p = np.square(sig - ref)
    return 10 * np.mean(np.log10(ref_p) - np.log10(noi_p))



def loadAudio(path, SR):
    data, sr = sf.read(path)
    try:
        assert sr == SR
    except AssertionError:
        data = librosa.resample(data, sr, opt.SR)
    return data - np.mean(data)

def eval(model, cleanPath, noisePath, opt):
    clean = loadAudio(cleanPath, opt.SR)
    noise = loadAudio(noisePath, opt.SR)
    noise = np.tile(noise, clean.shape[0] // noise.shape[0] + 1)[:clean.shape[0]]
    noiseAmp = np.mean(np.square(clean)) / np.power(10, opt.snr / 10.0)
    scale = np.sqrt(noiseAmp / np.mean(np.square(noise)))
    mix = clean + scale * noise
    leng = opt.nfft + (opt.nFrames - 1) * opt.hop
    results = np.array([])
    for i in range(leng, clean.shape[0], opt.hop):
        target = clean[i-leng:i][6*opt.hop:6*opt.hop+opt.nfft]
        input_ = {'A' : torch.from_numpy(mix[i-leng:i][np.newaxis, :]).float()}
        model.set_input(input_)
        model.test()
        output = model.get_current_visuals().data.cpu().numpy()
        output = output.flatten()
        print('output', output.shape)
        results = np.concatenate((results, output))
        print('snr', CalSNR(target, output))
    sf.write("results.wav", results, opt.SR)


opt = TestOptions().parse()
model = create_model(opt)

cleanPath = "/home/diggerdu/dataset/men/clean/p232_003.wav"
noisePath = "/home/diggerdu/dataset/men/noise/127118-saschaburghard-smallconcertaudienc-200.wav"


eval(model, cleanPath, noisePath, opt)

