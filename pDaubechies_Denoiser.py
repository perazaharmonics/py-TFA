import numpy as np
import cupy as cp
import pywt
import scipy.io.wavfile as wav

class DaubechiesWaveletDenoiser:
    def __init__(self, wavelet='db1', threshold_method='soft', threshold_value=0.1):
        self.wavelet = wavelet
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def denoise(self, signal):
        # Compute wavelet decomposition using PyWavelets on the CPU
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet)
        
        # Transfer coefficients to the GPU and apply thresholding
        thresholded_coeffs = [self._threshold(cp.array(c)) for c in coeffs]
        
        # Transfer thresholded coefficients back to the CPU for wavelet reconstruction
        thresholded_coeffs_cpu = [cp.asnumpy(c) for c in thresholded_coeffs]
        
        denoised_signal = pywt.waverec(thresholded_coeffs_cpu, wavelet=self.wavelet)
        return denoised_signal

    def _threshold(self, coeffs_gpu):
        # Perform thresholding on the GPU using CuPy
        if self.threshold_method == 'soft':
            return cp.maximum(coeffs_gpu - self.threshold_value, 0) * cp.sign(coeffs_gpu)
        elif self.threshold_method == 'hard':
            return coeffs_gpu * (cp.abs(coeffs_gpu) >= self.threshold_value)
        else:
            raise ValueError("Invalid thresholding method. Choose 'soft' or 'hard'.")
