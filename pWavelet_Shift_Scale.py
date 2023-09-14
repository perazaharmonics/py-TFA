import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Description: This script plots the wavelet function for different scaling factors and shifts.
Author: Josue E. Peraza Velazquez
Date: 14-09-2023

This scripts demonstrates the effect of scaling and shifting the wavelet function. A choice of common wavelets are given to the user. 
Most of these wavelets are applied in audio and speech processing, but they can be used in other applications as well.
In a previous implementation, a wavelet decomposition was performed on a signal using the QMF Filter Bank Structure H(Z) = H(-Z) 
to ensure perfect reconstruction (PR).
This script is a detailed view of some of the methods used for PR in the implementation in pMRA_Wavelets.py script.

'''


# Define the Haar wavelet function 

def haar(x, scaling_factor =1, shifting=0):
    # haar low-pass filter coefficients
    h = [1 / np.sqrt(2), 1 / np.sqrt(2)]  # low-pass filter coefficients
    # haar high-pass filter coefficients
    g = [1 / np.sqrt(2), - 1 / np.sqrt(2)]  # high-pass filter coefficients
    
    x_scaled = (x - shifting) / scaling_factor
    
    # Define the Haar wavelet function
    cond1 = ( 0 <= x_scaled) & (x_scaled < 1)
    cond2 = (1 <= x_scaled) & (x_scaled < 2)

    # Compute the Db2 wavelet values based on the conditions

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

    wavelet_values = np.where(
        cond1, 
        h[0] * np.sinc(x_scaled - 0) + h[1] * np.sinc(x_scaled - 1) ,
        np.where(
            cond2,
            h[0] * np.sinc(x_scaled - 1) + h[1] * np.sinc(x_scaled - 0) ,
            0.0
        )
    )
    
    return wavelet_values

def db6(x, scaling_factor=1, shifting=0):
    # db6 (D12) filter coefficients
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
    
    cond1 = (0 <= x_scaled) & (x_scaled < 0.5)
    cond2 = (0.5 <= x_scaled) & (x_scaled < 1)
    
    # Initialize the wavelet values array
    wavelet_values = np.zeros_like(x_scaled)

    # Interpolating my array of coefficients to make sure I have a smooth function
    for i in range(0, len(h)-1, 2): # Iterate two by two to make sure we don't exceed the array bounds
        term1 = h[i]*np.sinc(x_scaled - 2*i)
        term2 = h[i+1]*np.sinc(x_scaled - 2*i+1)
        
        values_cond1 = term1 + term2
        values_cond2 = h[i]*np.sinc(x_scaled - 2*i-1) + h[i+1]*np.sinc(x_scaled - 2*i+1)
        
        wavelet_values += np.where(cond1, values_cond1, np.where(cond2, values_cond2, 0.0))
    
    return wavelet_values


def bior15(x, scaling_factor=1, shifting=0):
    # Define Low-Pass filter coefficients for the bi-orthogonal 1.5 wavelet
    h = [-0.1294, 0.2241, 0.8365, 0.4830]

    # Define the High-Pass filter branch of the bi-orthogonal 1.5 wavelet filter bank
    g = [0.4830, -0.8395, 0.2241, 0.1294]

    x_scaled = (x - shifting) / scaling_factor
    
    wavelet_values = np.zeros_like(x_scaled)

    # Interpolating my array of coefficients to make sure I have a smooth function
    for i in range(len(h)):
        wavelet_values += h[i] * np.sinc(x_scaled - i)
    
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

functions = {
    '1': haar,
    '2': db1,
    '3': db6,
    '4': bior15
}
wavelet_function = functions.get(wavelet_choice, haar)

scaling_factors = [1, 2, 4]
shifts = [-2, -1, 0, 1, 2]

fig, ax = plt.subplots(len(scaling_factors), len(shifts), figsize=(10, 6))

for i, scaling_factor in enumerate(scaling_factors):
    for j, shifting in enumerate(shifts):
        y = wavelet_function(x, scaling_factor=scaling_factor, shifting=shifting)
        ax[i, j].plot(x, y)
        ax[i, j].set_title(f'Scale={scaling_factor}, Shift={shifting}')
        ax[i, j].grid(True)

plt.tight_layout()
plt.show()