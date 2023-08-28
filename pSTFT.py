import numpy as np
import matplotlib.pyplot as plt
def stft(x, fs, win, hop):
    w = np.hanning(win)
    out = np.array([np.fft.fft(w*x[i:i+win]) for i in range(0, len(x)-win, hop)])
    return out
fs = 44100  # for example
x = np.random.randn(fs*1)  # 1 second of random noise
X = stft(x, fs, 1024, 512)
plt.imshow(np.abs(X.T), aspect='auto', origin='lower')
plt.title('STFT Magnitude')
plt.ylabel('Frequency Bin')
plt.xlabel('Time Frame')
plt.colorbar()
plt.show()

"""
Note: There are advanced libraries like scipy and librosa in Python that provide more efficient and comprehensive implementations of STFT and related functions.
It's important to also discuss the resolution trade-offs associated with STFT (i.e., choosing window length affects time and frequency resolution)
and the concept of the inverse STFT (iSTFT, i.e. frequency synthesis) which can reconstruct a signal from its STFT representation.


"""
