import numpy as np
import cupy as cp
import pywt
from cupyx.scipy.signal import convolve  # using CuPy's version of convolve
import scipy.io.wavfile as wav

class MorletWaveletDenoiser:
    def __init__(self, scales=None, threshold_method='soft', threshold_value=0.1):
        self.scales = scales if scales else np.arange(1, 128)
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def denoise(self, input_signal):
        # Compute the Continuous Wavelet Transform (CWT) on the CPU
        coeffs, _ = pywt.cwt(input_signal, scales=self.scales, wavelet='cmor1.5-1.0')

        # Move data to GPU
        coeffs_gpu = cp.array(coeffs)

        # Threshold the wavelet coefficients on the GPU
        thresholded_coeffs_gpu = self._threshold(coeffs_gpu)

        # Reconstruct the denoised signal on the GPU
        denoised_signal_gpu = self._inverse_cwt_morlet(thresholded_coeffs_gpu)

        return cp.asnumpy(denoised_signal_gpu)  # transfer data back to CPU

    def _threshold(self, coeffs_gpu):
        if self.threshold_method == 'soft':
            return pywt.threshold(cp.asnumpy(coeffs_gpu), self.threshold_value, mode='soft')
        elif self.threshold_method == 'hard':
            return pywt.threshold(cp.asnumpy(coeffs_gpu), self.threshold_value, mode='hard')
        else:
            raise ValueError("Invalid thresholding method. Choose 'soft' or 'hard'.")

    def _inverse_cwt_morlet(self, coeffs_gpu):
        # Implement an inverse transformation specific to the Morlet wavelet on the GPU
        reconstruction_gpu = cp.zeros_like(coeffs_gpu[0])
        for scale, coeff_gpu in zip(self.scales, coeffs_gpu):
            wavelet_function_gpu = self._morlet(len(coeff_gpu), w=5, s=scale)
            reconstruction_gpu += convolve(coeff_gpu, wavelet_function_gpu, mode='same')

        return reconstruction_gpu / len(self.scales)  # Normalize by the number of scales

    def _morlet(self, M, w=5, s=1):
        # Custom morlet function to run on the GPU
        x = cp.linspace(-1, 1, M) * 2 * cp.pi
        output = cp.exp(1j * w * x)
        output *= cp.exp(-x**2 / (2 * s**2))  # Gaussian envelope
        return output
