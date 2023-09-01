import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


######################################################################
## Discrete Wavelet Transform
## Description: The Daubechies wavelets are a family of orthogonal wavelets characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## For the purpose of this discussion, let's look at the simplest of Daubechies wavelets: the D4, often referred to as db1. The D4 wavelet has two vanishing moments, which makes it optimal for representing piecewise constant functions.
## Low-Pass: The D4 wavelet is defined by four coefficients , h0, h1, h2, and h3. 
## High-Pass: the D4 wavelet is defined by the same coefficients, but in bit-reversed order and alternating signs: g3, g2, g1, and g0.
## Conclusion: These coefficients are used in a filter bank to either decompose a signal into approximation and detail coefficients (analysis) or reconstruct a signal from approximation and detail coefficients (synthesis).
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
######################################################################


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


def morlet(t, omega0=6):
    return np.pi**(-0.25) * np.exp(1j * omega0 * t) * np.exp(-0.5 * t**2)

def meyer(t):
    # Simplified Meyer wavelet function (not complete)
    pass  # Define your function here
def mexican_hat(t):
    return (1 - t**2) * np.exp(-t**2 / 2)

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


# Example
fs = 44100  # Sampling rate
t = np.linspace(0, 1, fs)  # Time vector
freq = 5  # Frequency in Hz
signal = np.sin(2 * np.pi * freq * t)  # Sine wave
# Perform the wavelet transform using your functions
approx, detail = db1(signal)  # Replace db1 with haar for Haar wavelet

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(t, signal)

plt.subplot(3, 1, 2)
plt.title("Approximation Coefficients")
plt.plot(approx)

plt.subplot(3, 1, 3)
plt.title("Detail Coefficients")
plt.plot(detail)

plt.tight_layout()
plt.show()

# Generate the wavelet
t_wavelet = np.linspace(-1, 1, fs)
morlet_wave = morlet(t_wavelet)  # Replace with meyer or mexican_hat for those wavelets

# Plotting
plt.figure(figsize=(12, 4))

plt.title("Morlet Wavelet")
plt.plot(t_wavelet, np.real(morlet_wave), label='Real part')
plt.plot(t_wavelet, np.imag(morlet_wave), label='Imaginary part')

plt.legend()
plt.show()
