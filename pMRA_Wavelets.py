import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.fftpack import ifft, fftshift
from scipy.interpolate import interp1d


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

#######################################################################################
## Synthesis
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################
def idwt_multilevel(coeffs, wavelet_func):
    signal = coeffs[0][0]  # Get the approximation coefficients of the highest level
    
    for i in range(len(coeffs) - 1, 0, -1):
        approx, detail = coeffs[i]
        signal = wavelet_func(approx, detail, signal)
    
    return signal

#######################################################################################
## Synthesis Haar
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################


def inverse_haar(approx, detail, previous_level):
    # Haar inverse transform
    reconstructed_signal = []
    for a, d in zip(approx, detail):
        reconstructed_signal.extend([(a + d) / 2, (a - d) / 2])
    return reconstructed_signal

#######################################################################################
## Synthesis db1
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################

def inverse_db1(approx, detail, previous_level):
    # Daubechies (db1) inverse transform
    reconstructed_signal = []
    for a, d in zip(approx, detail):
        reconstructed_signal.append(a + d)
        reconstructed_signal.append(previous_level[-1])
    return reconstructed_signal

#######################################################################################
## Synthesis db6
## Description: The synthesis filter bank is the inverse of the analysis filter bank. It is used to reconstruct a signal from approximation and detail coefficients.
## Reference: https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html
#######################################################################################

def inverse_db6(approx, detail):
    # Your db6 (D12) filter coefficients
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
    
    # Reverse the coefficients to get synthesis filters
    # Define Low-Low Filter
    h_inv = h[::-1]
    # Define High-Low Filter (bit reveral and alternating signs)
    g_inv = [-1 * coeff if idx % 2 else coeff for idx, coeff in enumerate(h_inv)]

    N = len(h_inv)
    reconstructed_signal = [0.0] * (2 * len(approx))  # Initialize with zeros

    # Loop through the approximation and detail coefficients
    for i in range(len(approx)):
        for k in range(N):
            index1 = (2 * i + k) % len(reconstructed_signal)
            index2 = (2 * i - k) % len(reconstructed_signal)
            
            reconstructed_signal[index1] += approx[i] * h_inv[k] + detail[i] * g_inv[k]
            reconstructed_signal[index2] += approx[i] * h_inv[k] - detail[i] * g_inv[k]

    return reconstructed_signal


# Synthesis Implementation
def synthesis(coeffs, wavelet_func):
    reconstructed_signal = idwt_multilevel(coeffs, wavelet_func)
    return reconstructed_signal

def generate_time_vector(original_t, target_length):
    interpolator = interp1d(np.linspace(0, 1, len(original_t)), original_t)
    new_t = interpolator(np.linspace(0, 1, target_length))
    return new_t


######################################################################
## UI Block
## Description: The following code snippet demonstrates the use of the DWT and CWT on a simple sine wave.
######################################################################

import matplotlib.pyplot as plt
import numpy as np

def plot_scalogram(coeffs):
    fig, ax = plt.subplots()
    
    # The vertical extent of the plot depends on the number of decomposition levels
    num_levels = len(coeffs)
    
    # Initialize vertical placement of the first level
    vertical_offset = 0
    
    # Loop through each level to plot approximation and detail coefficients
    for level, (approx, detail) in enumerate(coeffs):
        
        # The 'extent' sets the left, right, bottom, and top edges of the image
        extent = [0, len(approx), vertical_offset, vertical_offset + 1]
        vertical_offset += 1  # Move up for the next level
        
        # We plot the detail coefficients at this level
        ax.imshow(np.abs([detail]), aspect='auto', interpolation='nearest', cmap='jet', extent=extent)
        
        # Adjust vertical placement for approximation coefficients
        extent = [0, len(approx), vertical_offset, vertical_offset + 1]
        vertical_offset += 1  # Move up for the next level
        
        # We plot the approximation coefficients at this level
        ax.imshow(np.abs([approx]), aspect='auto', interpolation='nearest', cmap='jet', extent=extent)
    
    ax.set_title('Discrete Time-Frequency Representation')
    ax.set_xlabel('Time')
    ax.set_ylabel('Level (Scale)')
    plt.show()

# Example usage:
# coeffs = dwt_multilevel(signal, haar, 4)  # Use your dwt_multilevel function
# plot_scalogram(coeffs)




# Select wavelet from user input
def select_wavelet():
    print("Please select the DWT type:")
    print("1: Haar")
    print("2: db1 (Daubechies 4)")
    print("3: db6 (Daubechies 12)")
    choice = input("Enter the number corresponding to your choice: ")
    return choice




######################################################################
## Implementation
## Description: The following code snippet demonstrates the use of the DWT and CWT on a simple sine wave.
######################################################################
# Generate time vector


wavelet_choice = select_wavelet()
freq = 22050


# Define Discrete Input Signal
duration = 1
fs = 2 * freq

t = np.linspace(0, duration, fs*duration)

# Generate the signal dependent on the sampling frequency
signal = np.cumsum(np.random.randn(len(t)))

# Pad the signal to a power of 2
padded_signal, original_length = pad_to_pow2(signal)



# Pad the time vector
n = len(signal)
n_pad = int(next_power_of_2(n))
pad_values = n_pad - n
mean_dt = np.mean(np.diff(t))
padded_t = np.pad(t, (0, pad_values), 'constant', constant_values=(t[-1] + mean_dt))

# Replace the original signal and time vector with the padded versions
signal = padded_signal
t = padded_t

levels = 9  # Number of levels in the DWT  
coeffs = dwt_multilevel(signal, haar if wavelet_choice == '1' else db1 if wavelet_choice == '2' else db6, levels)

# Initialize the original time vector t with the length of the signal
tvec = [t]

# Generate the time vectors at each level of the DWT
for approx, detail in coeffs:
    target_length = len(approx)  # or len(detail), they should be the same
    new_t = generate_time_vector(t, target_length)
    tvec.append(new_t)


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


reconstructed_signal = synthesis(coeffs, inverse_haar if wavelet_choice == '1' else inverse_db1 if wavelet_choice == '2' else inverse_haar)

reconstructed_signal = reconstructed_signal[:original_length]


##########################################################
# Plotting for DWT
##########################################################
# Generate the time vectors at each level of the DWT
for i in range(1, levels + 1):
    tvec.append(generate_time_vector(t, i))

# Now, you can use these time_vectors while plotting
plt.figure(figsize=(12, 8))

# Plot the original signal
plt.subplot(2 * levels + 1, 1, 1)
plt.title("Original Signal")
plt.plot(tvec[0], signal)

# Plot the coefficients
for i, (approx, detail) in enumerate(coeffs):
    stretched_t = tvec[i + 1]  # Using the appropriate time vector

    plt.subplot(2 * levels + 1, 1, 2 * i + 2)
    plt.title(f"Level {i+1} Approximation Coefficients")
    plt.plot(stretched_t, approx)

    plt.subplot(2 * levels + 1, 1, 2 * i + 3)
    plt.title(f"Level {i+1} Detail Coefficients")
    plt.plot(stretched_t, detail)

plt.tight_layout()
plt.show()
plt.figure(figsize=(14, 8))

# Plot original signal
plt.subplot(3, 1, 1)
plt.title('Original Signal')
plt.plot(t, signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Use the last time_vector for the last-level approximation and detail coefficients
last_tvec = tvec[-1]
t_approx = generate_time_vector(original_t=t, target_length=len(approx))
t_detail = generate_time_vector(original_t=t, target_length=len(detail))
t_recon = generate_time_vector(original_t=t, target_length=len(reconstructed_signal))
# Plot Approximation Coefficients
plt.subplot(3, 1, 2)
plt.title('Approximation Coefficients')
plt.plot(t_approx, approx)
plt.xlabel('Time')
plt.ylabel('Amplitude')



# Plot Detail Coefficients
plt.subplot(3, 1, 3)
plt.title('Detail Coefficients')
plt.plot(t_detail, detail)
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Plot the original signal
plt.figure(figsize=(10, 5))
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Plot the reconstructed signal
plt.figure(figsize=(10, 5))
plt.plot( t_recon, reconstructed_signal)
plt.title('Reconstructed Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

plot_scalogram(coeffs)