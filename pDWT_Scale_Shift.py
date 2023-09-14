import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fftpack import ifft, fftshift
from scipy.interpolate import interp1d


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(np.ceil(np.log2(x)))

def pad_to_pow2(signal):
    original_length = len(signal)
    padded_length = int(next_power_of_2(original_length))
    padding_zeros = padded_length - original_length
    padded_signal = np.pad(signal, (0, padding_zeros), mode='symmetric')
    return padded_signal, original_length

def remove_padding(signal, original_length):
    return signal[:original_length]

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


def haar(x, scaling_factor =1, shifting=0):
    # haar low-pass filter coefficients
    h = [1 / np.sqrt(2), 1 / np.sqrt(2)]  # low-pass filter coefficients
    # haar high-pass filter coefficients
    g = [1 / np.sqrt(2), - 1 / np.sqrt(2)]  # high-pass filter coefficients
    
    
    # Initialize
    approx = []
    detail = []

    

def haar(x, scaling_factor =1, shifting=0):
    # haar low-pass filter coefficients
    h = [1 / np.sqrt(2), 1 / np.sqrt(2)]  # low-pass filter coefficients
    # haar high-pass filter coefficients
    g = [1 / np.sqrt(2), - 1 / np.sqrt(2)]  # high-pass filter coefficients
    
    x_scaled = (x - shifting) / scaling_factor
    
    # Define the Haar wavelet function
    cond1 = ( 0 <= x_scaled) & (x_scaled < 0.5)
    cond2 = (0.5 <= x_scaled) & (x_scaled < 1)

    # Compute the Haar wavelet values based on the conditions
    wavelet_values = np.where(cond1, 1.0, np.where(cond2, -1.0, 0.0))
    
    return wavelet_values


def db1(x, scaling_factor =1, shifting=0):

    # db1 (D4) filter coefficients
    h = [(1 + np.sqrt(3))/4, (3 + np.sqrt(3))/4, (3 - np.sqrt(3))/4, (1 - np.sqrt(3))/4]  # low-pass filter coefficients
    g = [h[3], -h[2], h[1], -h[0]]  # high-pass filter coefficients

    x_scaled = (x - shifting) / scaling_factor
    
    # Define the Haar wavelet function
    cond1 = ( 0 <= x_scaled) & (x_scaled < 0.5)
    cond2 = (0.5 <= x_scaled) & (x_scaled < 1)

    # Compute the Haar wavelet values based on the conditions
    wavelet_values = np.where(cond1, 1.0, np.where(cond2, -1.0, 0.0))
    
    return wavelet_values

def db6(x, scaling_factor =1, shifting=0):
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


    x_scaled = (x - shifting) / scaling_factor
    
    # Define the Haar wavelet function
    cond1 = ( 0 <= x_scaled) & (x_scaled < 0.5)
    cond2 = (0.5 <= x_scaled) & (x_scaled < 1)

    # Compute the Haar wavelet values based on the conditions
    wavelet_values = np.where(cond1, 1.0, np.where(cond2, -1.0, 0.0))
    
    return wavelet_values


def bior15(x, scaling_factor =1, shifting=0):
    # Define Low-Pass filter coefficients for the bi-orthogonal 1.5 wavelet
    h = [-0.1294, 0.2241, 0.8365, 0.4830]

    # Define the High-Pass filter branch of the bi-orthogonal 1.5 wavelet filter bank
    g = [0.4830, -0.8395, 0.2241, 0.1294]


    # Get number of samples
    N = len(h)

    # Get the different scales of the wavelet per orthogonal shift
    x_scaled = (x - shifting) / scaling_factor
    
    # Define the Haar wavelet function
    cond1 = ( 0 <= x_scaled) & (x_scaled < 0.5)
    cond2 = (0.5 <= x_scaled) & (x_scaled < 1)

    # Compute the Haar wavelet values based on the conditions
    wavelet_values = np.where(cond1, 1.0, np.where(cond2, -1.0, 0.0))
    
    return wavelet_values

# Select wavelet from user input
def select_wavelet():
    print("Please select the DWT type:")
    print("1: Haar")
    print("2: db1 (Daubechies 4)")
    print("3: db6 (Daubechies 12)")
    print("4: bior15")

    choice = input("Enter the number corresponding to your choice: ")
    return choice

wavelet_choice = select_wavelet()
x = np.linspace(-2, 2, 1000)

if wavelet_choice == '1':
    y = haar(x, scaling_factor=1, shifting = 0)  # Haar wavelet
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Haar Wavelet')
    plt.title('Haar Wavelet with Scaling Factor 1 and Shifting 0')
    plt.grid(True)
    plt.show()

elif wavelet_choice == '2':
    y = db1(x, scaling_factor=1, shifting = 0)  # db1 (Daubechies 4) wavelet
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Daubechies 4 Wavelet')
    plt.title('Daubechies 4 Wavelet with Scaling Factor 1 and Shifting 0')
    plt.grid(True)
    plt.show()

elif wavelet_choice == '3':
    y = db6(x, scaling_factor=1, shifting = 0)  # db6 (Daubechies 12) wavelet
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Daubechies 12 Wavelet')
    plt.title('Daubechies 12 Wavelet with Scaling Factor 1 and Shifting 0')
    plt.grid(True)
    plt.show()
elif wavelet_choice == '4':
    y = bior15(x, scaling_factor=1, shifting = 0)  # bi-orthogonal 15 wavelet
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Bi-Orthogonal 15 Wavelet')
    plt.title('Bi-Orthogonal 15 Wavelet with Scaling Factor 1 and Shifting 0')
    plt.grid(True)
    plt.show()
else: 
    print("Invalid choice. Using Haar as default.")
    y = haar(x, scaling_factor=1, shifting = 0)  # Haar wavelet
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Haar Wavelet')
    plt.title('Haar Wavelet with Scaling Factor 1 and Shifting 0')
    plt.grid(True)
    plt.show()


