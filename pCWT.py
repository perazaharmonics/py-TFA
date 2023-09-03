import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fftpack import ifft, fftshift

######################################################################
## Discrete Wavelet Transform
## Description: The Daubechies wavelets are a family of orthogonal wavelets characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## For the purpose of this discussion, let's look at the simplest of Daubechies wavelets: the D4, often referred to as db1. The D4 wavelet has two vanishing moments, which makes it optimal for representing piecewise constant functions.
## Low-Pass: The D4 wavelet is defined by four coefficients , h0, h1, h2, and h3. 
## High-Pass: the D4 wavelet is defined by the same coefficients, but in bit-reversed order and alternating signs: g3, g2, g1, and g0.
## Conclusion: These coefficients are used in a filter bank to either decompose a signal into approximation and detail coefficients (analysis) or reconstruct a signal from approximation and detail coefficients (synthesis).
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
######################################################################

######################################################################
## Discrete Wavelet Transform
## Description: The Discrete Wavelet Transform (DWT) is a time-frequency representation of signals. 
# It uses a filter bank to decompose a signal into approximation and detail coefficients. The DWT is computed by convolving the signal with the low-pass and high-pass filters, and then downsampling the result.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
######################################################################

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(np.ceil(np.log2(x)))

 # Scales the coeeficients to the range [0, 1]
def normalize_minmax(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

# Scales the coeeficients to have zero mean and unit variance
def normalize_zscore(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val


######################################################################
## Continuous Wavelet Transform
## Description: The continuous wavelet transform (CWT) is a time-frequency representation of signals. It uses a kernel function, called the wavelet, which is scaled and translated over the signal. The CWT is computed as the inner product of the signal with the scaled and translated wavelet.
## Method: The CWT is computed by convolving the signal with the scaled and translated wavelet. The wavelet is scaled by multiplying it with a dilatation parameter a, and translated by adding a translation parameter b. The CWT is computed as the inner product of the signal with the scaled and translated wavelet.
######################################################################

# Continuous Wavelet Transform (CWT) Algorithm
def cwt(signal, wavelet_function, scales):
    coefficients = []
    for scale in scales:
        wavelet_data = wavelet_function(np.linspace(-1, 1, len(signal)), scale)
        coef = convolve(signal, wavelet_data, mode='same')
        coefficients.append(coef)
    return coefficients



def morlet(t, omega0=6):
    return np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-0.5 * t**2)


def meyer_wavelet_fourier(omega):
    result = np.zeros_like(omega)
    mask1 = (omega >= np.pi) & (omega <= 2 * np.pi)
    mask2 = (omega >= 2 * np.pi) & (omega <= 4 * np.pi)
    result[mask1] = np.sin(np.pi / 2 * np.tanh((omega[mask1] - np.pi) / (np.pi - 2)))
    result[mask2] = np.cos(np.pi / 2 * np.tanh((omega[mask2] - 2 * np.pi) / (2 * np.pi - 3)))
    return result

def meyer_wavelet_time(N):
    omega = 2 * np.pi * np.fft.fftfreq(N)
    meyer_fourier = meyer_wavelet_fourier(omega)
    meyer_time = fftshift(ifft(meyer_fourier).real)
    return meyer_time

def meyer(t, scale):
    N = len(t)
    meyer_t = meyer_wavelet_time(N)
    return np.interp(t / scale, t, meyer_t)

######################################################################
## Define Ricker Wavelet (mexican hat)
######################################################################
def mexican_hat(t):
    return (1 - t**2) * np.exp(-t**2 / 2)


######################################################################
## Plot Scalogram
## Description: A scalogram is a visual representation of the wavelet transform. 
# It is a plot of the magnitude of the wavelet coefficients as a function of time and scale.
# Reference: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
######################################################################
def plot_scalogram_and_detail(coefficients, wavelet_name, scales, t):
    # Plot the Scalogram
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, 1, min(scales), max(scales)], cmap='jet')
    plt.colorbar(label="Magnitude")
    plt.title(f"{wavelet_name} Wavelet Transform (CWT)")
    plt.xlabel("Time")
    plt.ylabel("Scale")

    # Plot the detail plot
    plt.subplot(2, 1, 2)
    plt.title(f"{wavelet_name} Wavelet Transform (CWT) Detail Coefficients")
    for scale_idx, scale in enumerate(scales):
        plt.plot(t, coefficients[scale_idx] + scale_idx * 10)
        plt.xlabel("Time")
        plt.ylabel("Detail Coefficients")
    
    plt.tight_layout()
    plt.show()



def select_wavelet():
    print("Please select the wavelet type:")
    print("1: Mexican Hat")
    print("2: Meyer")
    print("3: Morlet")
    choice = input("Enter the number corresponding to your choice: ")
    return choice



######################################################################
## Implementation
## Description: The following code snippet demonstrates the use of the DWT and CWT on a simple sine wave.
######################################################################

# Select wavelet from user input
wavelet_choice = select_wavelet()
freq = 22050
# Processing according to user input
if wavelet_choice in ['1', '2', '3']:
# Define Continuous Input Signal
    freq = 22050  # Frequency in Hz
    t = np.linspace(0, 1, freq)  # Time vector
    signal = np.cumsum(np.random.randn(len(t)))

    # Define scales
    scales = 2**np.arange(4, 14)

    # Perform the CWT
    # Lambda expressions to ensure same function signature
    if wavelet_choice == '1':
        wavelet_choice = 'Ricker (Mexican Hat))'
        wavelet_function = lambda t, scale: mexican_hat(t)
    elif wavelet_choice == '2':
        wavelet_choice = 'Meyer'
        wavelet_function = meyer  # already matches the expected function signature
   
    elif wavelet_choice == '3':
        wavelet_choice = 'Morlet'
        wavelet_function = lambda t, scale: morlet(t, omega0=12)
    else:
        print("Invalid choice. Using Mexican Hat as default.")
        wavelet_function = lambda t, scale: mexican_hat(t)

    coefficients = cwt(signal, wavelet_function, scales)
    # Normalize the coefficients
    normalized_coefficients = normalize_zscore(coefficients)


    # Plot Scalogram
    plot_scalogram_and_detail(coefficients, wavelet_choice, scales, t)

    