#将声音信号转变为
from pandas import set_eng_float_format
import torch
from torch import stft
import torchaudio

stft_function=torchaudio.transforms.Spectrogram(n_fft=35,win_length=12,hop_length=32)

wave=torch.rand(1,1600)

stft_spectrum=stft_function(wave)

print(stft_spectrum.shape)