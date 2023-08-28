import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd


######################################################################
## Discrete Wavelet Transform
## Description: The Daubechies wavelets are a family of orthogonal wavelets characterized by a maximal number of vanishing moments for some given support. In simpler terms, this means that the wavelets are designed to represent polynomial data of a given order efficiently.
## For the purpose of this discussion, let's look at the simplest of Daubechies wavelets: the D4, often referred to as db1. The D4 wavelet has two vanishing moments, which makes it optimal for representing piecewise constant functions.
## Low-Pass: The D4 wavelet is defined by four coefficients , h0, h1, h2, and h3. 
## High-Pass: the D4 wavelet is defined by the same coefficients, but in bit-reversed order and alternating signs: g3, g2, g1, and g0.
## Conclusion: These coefficients are used in a filter bank to either decompose a signal into approximation and detail coefficients (analysis) or reconstruct a signal from approximation and detail coefficients (synthesis).
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
######################################################################


def dwt(signal):

    # db1 (D4) filter coefficients
    h = [(1 +np.sqrt(3)/4), (3 + np.sqrt(3))/4, (3 - np.sqrt(3))/4, (1 - np.sqrt(3))/4] # low-pass filter coefficients
    g [h[3], -h[2], h[1], -h[0]] # high-pass filter coefficients

    # Decompose signal into approximation (Low-Pass) and detail (High-Pass) coefficients

    # Low-Pass: Approximation coefficients (Low-frequency resolution, high temporal resolution )
    approx = [ sum(h[k]*signal[i*2 - k] for k in range(4)) for i 
              in range(len(signal)//2) ]
    #High-Pass: Detail coefficients (High-frequency resolution, low temporal resolution)
    detail = [ sum(g[k]*signal[i*2 - k] for k in range(4)) for i 
              in range(len(signal)//2) ]
    
    return approx, detail

# Example
signal = [1, 2, 3, 4, 5, 6, 7, 8]
approx, detail = dwt(signal)
# Plot the original signal, approximation, and detail coefficients
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(signal, marker='o')
plt.title('Original Signal')

plt.subplot(3, 1, 2)
plt.plot(approx, marker='o', color='r')
plt.title('Approximation Coefficients')

plt.subplot(3, 1, 3)
plt.plot(detail, marker='o', color='g')
plt.title('Detail Coefficients')

plt.tight_layout()
plt.show()