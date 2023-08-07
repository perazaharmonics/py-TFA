import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import warnings
from pNLM_Denoising import NLM_Denoiser

# Load audio file
Fs, y = wav.read('Test_BB.wav')
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
    print("\n Starting denoising process...")
    y_nlm = denoiser.denoise(y)
    print("\n Denoising process completed.")

    # Time-domain plot for original signal
    plt.figure()
    plt.plot(y, 'b')
    plt.title(f'Original Signal (search_window={search_window}, patch_size={patch_size}, h={h})')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend(['Original Signal'])
    plt.show()

    # Time-domain plot for denoised signal
    plt.figure()
    plt.plot(y_nlm, colors[i])
    plt.title(f'Non-Local Means Denoised Signal (search_window={search_window}, patch_size={patch_size}, h={h})')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend(['Non-Local Means Denoised Signal'])
    plt.show()

   # Calculate magnitude spectra for original and denoised signals
    N = len(y)
    Y = fft(y)[:N//2]
    Y_nlm = fft(y_nlm)[:N//2]
    f = np.linspace(0, Fs/2, N//2)

    # Magnitude spectra plot for original signal
    plt.figure()
    plt.semilogx(f, 20 * np.log10(np.abs(Y)), 'b')
    plt.title(f'Original Signal Magnitude Spectra (search_window={search_window}, patch_size={patch_size}, h={h})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Original Signal'])
    plt.show()

    # Magnitude spectra plot for denoised signal
    plt.figure()
    plt.semilogx(f, 20 * np.log10(np.abs(Y_nlm)), colors[i])
    plt.title(f'Non-Local Means Denoised Signal Magnitude Spectra (search_window={search_window}, patch_size={patch_size}, h={h})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend(['Non-Local Means Denoised Signal'])
    plt.show()