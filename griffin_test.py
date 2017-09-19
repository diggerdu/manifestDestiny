from util.audio_process import recovery_phase
import soundfile as sf
import librosa
import numpy as np


sig, sr = sf.read("origin.wav")

NFFT = 512
HOP = 128
iters = 2000

amp = librosa.stft(sig, n_fft=NFFT, hop_length=HOP)
raw = librosa.istft(amp, win_length=NFFT, hop_length=HOP)

sf.write('reconstruct.wav', raw, sr)


'''
phase = 2 * np.pi *np.random.random_sample(amp.shape) - np.pi
for i in range(1001, iters):
    spec = amp * np.exp(1j * phase)
    raw = librosa.istft(spec, win_length=NFFT, hop_length=HOP)
    if i % 20 == 0:
        sf.write('results/' + str(i) + 'epoch.wav', raw, sr)
    phase = np.angle(librosa.stft(raw, n_fft=NFFT, hop_length=HOP))

print(amp.shape)
'''
