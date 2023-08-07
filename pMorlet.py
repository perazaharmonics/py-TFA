# MorletWaveletDenoiser.py
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import sounddevice as sd
from pMorlet_Denoiser import MorletWaveletDenoiser

# Function to denoise signal
def denoise_signal(denoiser, y):
    y_denoised = denoiser.denoise(y)
    return y_denoised

# Load audio file
print("Reading the file")
Fs, y = wav.read('BB_Anal2.wav')
y = y / np.max(np.abs(y))  # Normalizing

# Denoising
denoiser = MorletWaveletDenoiser()
y_denoised = denoise_signal(denoiser, y)

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
plt.title("Denoised Signal using Morlet Wavelet Denoiser")
plt.show()
