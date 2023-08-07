import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import sounddevice as sd
from pSymlet_Denoiser import SymletWaveletDenoiser

def process_in_segments(signal, segment_size, processing_func):
    segments = [signal[i:i+segment_size] for i in range(0, len(signal), segment_size)]
    processed_segments = [processing_func(segment) for segment in segments]
    return np.concatenate(processed_segments)

print("Reading the file")
Fs, y = wav.read('BB_Anal2.wav')
y = y / np.max(np.abs(y))  # Normalizing

# Denoising
denoiser = SymletWaveletDenoiser('sym2', 'soft', 0.1)
y_denoised = process_in_segments(y, segment_size=44100, processing_func=denoiser.denoise)

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
plt.title("Denoised Signal using Symlet Wavelet Denoiser")
plt.show()
