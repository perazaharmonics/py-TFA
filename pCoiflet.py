import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from pCoiflet_Denoiser import CoifletWaveletDenoiser

print("\n reading the file \n")
Fs, y = wav.read('BB_Anal2.wav')
y = y / np.max(np.abs(y))  # Normalizing
Y = fft(y)
f = np.linspace(0, Fs / 2, len(Y) // 2)

denoiser = CoifletWaveletDenoiser('coif1', 'soft', 0.1)  # You can change the parameters
y_denoised = denoiser.denoise(y)
Y_denoised = fft(y_denoised)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(y_denoised, 'r')
plt.title('Denoised Signal using CoifletWaveletDenoiser')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.subplot(2, 1, 2)
plt.semilogx(f, 20 * np.log10(np.abs(Y[:len(Y)//2]) + 1e-6), 'b')
plt.title('Magnitude Spectra after denoising with CoifletWaveletDenoiser')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.show()
