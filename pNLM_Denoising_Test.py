import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from pNLM_Denoising import NLM_Denoiser

# NLM_Denoiser class definition here (as previously provided)
# Make sure the pNLM_Denoising.py file containing the NLM_Denoiser class is in the same directory or in the PYTHONPATH

# Load audio file
Fs, y = wav.read('BB_Anal2.wav')
y = y / np.max(np.abs(y))  # Normalizing, assuming the signal is not already normalized

# Define different sets of Non-Local Means parameters
params = np.array([
    [5, 2, 0.4432],
    [10, 2, 0.5],  # Reduced patch_size from 3 to 2
    [20, 2, 1.0],  # Reduced patch_size from 5 to 2
])

# Define plot colors for each set of parameters
colors = ['r', 'g', 'm']

# Apply Non-Local Means denoising with different parameters and plot results
for i in range(params.shape[0]):
    search_window = params[i, 0]
    patch_size = params[i, 1]
    h = params[i, 2]
    
    denoiser = NLM_Denoiser(search_window=search_window, patch_size=patch_size, h=h)
    y_nlm = denoiser.denoise(y)

    # Time-domain comparison plot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(y, 'b')
    plt.plot(y_nlm, colors[i])
    plt.title(f'Time-domain Comparison (search_window={search_window}, patch_size={patch_size}, h={h})')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend(['Original Signal', 'Non-Local Means Denoised Signal'])
    
    # Magnitude spectra comparison plot
    plt.subplot(2, 1, 2)
    plt.semilogx(f, 20 * np.log10(Y), 'b')
    plt.semilogx(f, 20 * np.log10(Y_nlm), colors[i])
    plt.title('Magnitude Spectra')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Original Signal', 'Non-Local Means Denoised Signal'])
    plt.show()
