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

import numpy as np

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

def dwt_multilevel(signal, wavelet_func, levels):
    n = len(signal)
    n_pad = int(next_power_of_2(n))
    
    # Zero-padding to make the length a power of 2
    if n_pad != n:
        pad_values = n_pad - n
        signal = np.pad(signal, (0, pad_values), 'constant')
        
    coeffs = []
    current_signal = signal
    
    for i in range(levels):
        approx, detail = wavelet_func(current_signal)
        coeffs.append((approx, detail))
        current_signal = approx
        if len(current_signal) < 2:
            break  # Terminate if the signal length becomes less than 2
    
    return coeffs


def haar(signal):
    # haar low-pass filter coefficients
    h = [1 / np.sqrt(2), 1 / np.sqrt(2)]  # low-pass filter coefficients
    # haar high-pass filter coefficients
    g = [1 / np.sqrt(2), - 1 / np.sqrt(2)]  # high-pass filter coefficients
    
    
    # Initialize
    approx = []
    detail = []

    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = h[0]*signal[2*i] + h[1]*signal[2*i + 1]
        detail_sum = g[0]*signal[2*i] + g[1]*signal[2*i + 1]
        approx.append(approx_sum)
        detail.append(detail_sum)

    return approx, detail

def db1(signal):

    # db1 (D4) filter coefficients
    h = [(1 + np.sqrt(3))/4, (3 + np.sqrt(3))/4, (3 - np.sqrt(3))/4, (1 - np.sqrt(3))/4]  # low-pass filter coefficients
    g = [h[3], -h[2], h[1], -h[0]]  # high-pass filter coefficients

    # Initialize
    approx = []
    detail = []

    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(len(h)):
            index = 2 * i + k - len(h) // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)

    return approx, detail

def db6(signal):
    # db6 (D12) filter coefficients
    # db6 low-pass filter coefficients
    h = [
        -0.001077301085308,
        0.0047772575109455,
        0.0005538422011614,
        -0.031582039318486,
        0.027522865530305,
        0.097501605587322,
        -0.129766867567262,
        -0.226264693965440,
        0.315250351709198,
        0.751133908021095,
        0.494623890398453,
        0.111540743350109
    ]

    # db6 high-pass filter coefficients
    g = [
        h[11],
        -h[10],
        h[9],
        -h[8],
        h[7],
        -h[6],
        h[5],
        -h[4],
        h[3],
        -h[2],
        h[1],
        -h[0]
    ]


    

    # Initialize
    N = len(h)
    approx = []
    detail = []

    # Generate coefficients
    for i in range(0, len(signal) // 2):
        approx_sum = 0
        detail_sum = 0
        for k in range(N):
            index = 2 * i + k - N // 2 + 1
            if 0 <= index < len(signal):
                approx_sum += h[k] * signal[index]
                detail_sum += g[k] * signal[index]
        approx.append(approx_sum)
        detail.append(detail_sum)

    return approx, detail


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
## Plotting the Filter Bank Structure
######################################################################


def plot_filter_bank(h, g, fs):
    N = 512  # Number of points for FFT
    H = np.fft.fft(h, N)
    G = np.fft.fft(g, N)
    freqs = np.fft.fftfreq(N, 1/fs)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title("Low-pass Filter (h) Frequency Response")
    plt.plot(freqs[:N//2], np.abs(H)[:N//2])  # Plot only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(1, 2, 2)
    plt.title("High-pass Filter (g) Frequency Response")
    plt.plot(freqs[:N//2], np.abs(G)[:N//2])  # Plot only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

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
    print("1: Haar")
    print("2: db1 (Daubechies 4)")
    print("3: db6 (Daubechies 12)")
    print("4: Mexican Hat")
    print("5: Meyer")
    print("6: Morlet")
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
    # Define Discrete Input Signal
    duration = 1  
    fs = 2*freq  
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  
    
    signal = np.cumsum(np.random.randn(len(t)))
    
    n = len(signal)
    n_pad = int(next_power_of_2(n))
    pad_values = n_pad - n

    # Pad your signal and time vector
    signal = np.pad(signal, (0, pad_values), 'constant')
    t = np.pad(t, (0, pad_values), 'constant', constant_values=t[-1] + np.mean(np.diff(t)))
    
    levels = 9  # Number of levels in the DWT  
    coeffs = dwt_multilevel(signal, haar if wavelet_choice == '1' else db1 if wavelet_choice == '2' else db6, levels)
    

    ##########################################################
    ## Perform the DWT
    ##########################################################
    
    if wavelet_choice == '1':
        approx, detail = haar(signal)  # Haar wavelet
    elif wavelet_choice == '2':
        approx, detail = db1(signal)  # db1 (Daubechies 4) wavelet
    elif wavelet_choice == '3':
        approx, detail = db6(signal)  # db6 (Daubechies 12) wavelet
    else: 
        print("Invalid choice. Using Haar as default.")
        approx, detail = haar(signal)  # Haar wavelet
        
    ##########################################################
    # Plotting for DWT
    ##########################################################
            # Create a scalogram
    plt.figure(figsize=(12, 8))
    plt.subplot(2 * levels + 1, 1, 1)
    plt.title("Original Signal")
    plt.plot(t, signal)

    current_t = np.linspace(t[0], t[-1], len(t)//2)  # Initialize with halved time vector
    for i, (approx, detail) in enumerate(coeffs):
        stretched_t = np.linspace(t[0], t[-1], len(approx))  # Stretch the time vector to the original length
        
        plt.subplot(2 * levels + 1, 1, 2 * i + 2)
        plt.title(f"Level {i + 1} - Approximation Coefficients")
        plt.plot(stretched_t, approx)  # Plot against stretched time vector
        
        plt.subplot(2 * levels + 1, 1, 2 * i + 3)
        plt.title(f"Level {i + 1} - Detail Coefficients")
        plt.plot(stretched_t, detail)  # Plot against stretched time vector
        
        current_t = current_t[::2]  # Update for the next level

    plt.tight_layout()
    plt.show()
   
    plt.figure(figsize=(14, 8))

    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.title('Original Signal')
    plt.plot(t, signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot Approximation Coefficients
    plt.subplot(3, 1, 2)
    plt.title('Approximation Coefficients')
    plt.plot(approx)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    # Plot Detail Coefficients
    plt.subplot(3, 1, 3)
    plt.title('Detail Coefficients')
    plt.plot(detail)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    

    
else:
    # Define Continuous Input Signal
    freq = 22050  # Frequency in Hz
    t = np.linspace(0, 1, freq)  # Time vector
    signal = np.cumsum(np.random.randn(len(t)))

    # Define scales
    scales = 2**np.arange(4, 14)




    # Perform the CWT
    # Lambda expressions to ensure same function signature
    if wavelet_choice == '4':
        wavelet_choice = 'Ricker (Mexican Hat))'
        wavelet_function = lambda t, scale: mexican_hat(t)
    elif wavelet_choice == '5':
        wavelet_choice = 'Meyer'
        wavelet_function = meyer  # already matches the expected function signature
   
    elif wavelet_choice == '6':
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