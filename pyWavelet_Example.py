import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# Importing the denoiser classes
from pMorlet_Denoiser import MorletWaveletDenoiser
from pDaubechies_Denoiser import DaubechiesWaveletDenoiser
from pGaussian_Wavelet_Denoiser import GaussianWaveletDenoiser
from pCoiflet_Denoiser import CoifletWaveletDenoiser
from pSymlet_Denoiser import SymletWaveletDenoiser
from multiprocessing import Pool
# Load audio file
print(  " \n reading the file \n")


# Function to denoise signal
def denoise_signal(denoiser):
    y_denoised = denoiser.denoise(y)
    return y_denoised, type(denoiser).__name__

# Load audio file
print("loading aduio wav file")
Fs, y = wav.read('BB_Anal2.wav')
print("\n Normalizing the signal \n")
y = y / np.max(np.abs(y))  # Normalizing
# Get spectra(FFT)
Y = fft(y)
# Define vector frequncies space
print("\n Defining the vector frequencies space \n")
f = np.linspace(0, Fs / 2, len(Y) // 2)
print("\n initializing the denoisers \n")
# List of denoisers to iterate through
denoisers = [               
    MorletWaveletDenoiser(),
    DaubechiesWaveletDenoiser('db1', 'soft', 0.1),
    DaubechiesWaveletDenoiser('db10', 'hard', 0.01),
    GaussianWaveletDenoiser(),
    CoifletWaveletDenoiser('coif1', 'soft', 0.1),
    SymletWaveletDenoiser('sym2', 'soft', 0.1) 
    ]
print("\n denoising the signal \n")                                                                                               
results = [denoise_signal(denoiser) for denoiser in denoisers]

print("\n plotting the results \n")
    # Plotting the results
    
for y_denoised, denoiser_name in results:
        Y_denoised = fft(y_denoised)
        print(f'Plotting results for {denoiser_name}')
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(y_denoised, 'r')
        plt.title(f'Denoised Signal using {denoiser_name}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.semilogx(f, 20 * np.log10(np.abs(Y[:len(Y)//2]) + 1e-6), 'b')
        plt.title(f'Magnitude Spectra after denoising with {denoiser_name}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.show()