# GaussianWaveletDenoiser.py
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sounddevice as sd
from pGaussian_Wavelet_Denoiser import GaussianWaveletDenoiser

print("Reading the file")
Fs, y = wav.read('BB_Anal2.wav')
y = y / np.max(np.abs(y))  # Normalizing

# Denoising
denoiser = GaussianWaveletDenoiser()
y_denoised = denoiser.denoise(y)

# Plot and play
print("Playing original audio")
sd.play(y, samplerate=Fs)
sd.wait()
plt.plot(y)
plt.title("Original Signal")
plt.show()
print("Playing denoised audio")
sd.play(y_denoised, samplerate=Fs)
sd.wait()
plt.plot(y_denoised)
plt.title("Denoised Signal using Gaussian Wavelet Denoiser")
plt.show()
