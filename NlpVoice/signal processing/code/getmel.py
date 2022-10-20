#得到mel谱的示例代码
import librosa
import soundfile

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy.signal as signal
import copy

sr = 16000 # Sample rate.
n_fft = 1024 # fft points (samples)
frame_shift = 0.0125 # seconds  #时间表示法
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples. 采样点表示法
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate 梅尔谱的频率组数
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or预加重参数
max_db = 100
ref_db = 20
top_db = 15


def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=22050)  #y是一维的时域信号,sr是采样率

    # 画图
    # plt.figure()
    # plt.title("oringin: wavform")
    # plt.plot(y)
    # pp = PdfPages('临时.pdf')
    # plt.savefig(pp, format='pdf', bbox_inches='tight')
    # pp.close()
    # plt.show()

    # print(y.shape)
    # print(sr)


    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)#切除掉强度低于某个值的

    # print(y.shape)

    # Preemphasis  #预加重
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # print(y.shape)

    # stft  #直接得到短时傅里叶谱，幅度和相位是一个复数
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # print(linear) #这是一个复数

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # print(mag)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) #梅尔谱矩阵乘以幅度谱矩阵,注意这边中间结果的变化，矩阵行和列的含义

    # to decibel，取对数
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize，归一化，最小值和最大值都有限制
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    print(mel.shape)

    return mel, mag



if __name__ == '__main__':
    ## 给定一条语音
    p = '/home/jishengpeng/NlpVoice/getmel/0001_000351.wav'
    get_spectrograms(p)