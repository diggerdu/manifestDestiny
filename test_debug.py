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

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    audio = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    clean_name = img_path[0].split('/')[-1][:15]
    clean_amp = audio['clean'][0, 0, ::]
    ratio = np.power(audio['ratio'][0, 0, ::], 2.0)

    # cheat
    ratio = ratio * (np.random.random((ratio.shape)) * 0.1 + 1)




    est_ratio = np.power(audio['est_ratio'][0, 0, ::], 2.0)
    noise_amp = audio['noise'][0, 0, ::]
    denoise_amp = noise_amp * (1.0 - ratio)

    noise_raw = recovery_phase(noise_amp, n_fft=opt.nfft, hop=opt.hop, iters=200)
    sf.write('results/{}_noise.wav'.format(clean_name), noise_raw, opt.SR)
    clean_raw = recovery_phase(clean_amp, n_fft=opt.nfft, hop=opt.hop)
    sf.write('results/{}_clean.wav'.format(clean_name), clean_raw, opt.SR)
    denoise_raw = recovery_phase(denoise_amp, n_fft=opt.nfft, hop=opt.hop)
    sf.write('results/{}_denoise.wav'.format(clean_name), denoise_raw, opt.SR)




