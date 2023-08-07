import numpy as np
import cupy as cp
import pywt
import scipy.signal
import scipy.io.wavfile as wav

class GaussianWaveletDenoiser:
    def __init__(self, wavelet='gaus1', scales=None, threshold_method='soft', threshold_value=0.1):
        if not wavelet.startswith('gaus'):
            raise ValueError("Wavelet must be from the Gaussian family ('gaus1' to 'gaus8').")
        self.wavelet = wavelet
        self.scales = scales if scales else np.arange(1, 128)
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def denoise(self, signal):
        coefficients, _ = pywt.cwt(signal, scales=self.scales, wavelet=self.wavelet)
        
        # Transfer coefficients to the GPU and apply thresholding
        thresholded_coefficients = self._threshold(cp.array(coefficients))

        # Transfer thresholded coefficients back to the CPU for inverse CWT
        thresholded_coefficients_cpu = cp.asnumpy(thresholded_coefficients)
        
        denoised_signal = pywt.icwt(thresholded_coefficients_cpu, None, self.scales, wavelet=self.wavelet)
        return denoised_signal.ravel()

    def _threshold(self, coeffs_gpu):
        # Perform thresholding on the GPU using CuPy
        if self.threshold_method == 'soft':
            return cp.maximum(coeffs_gpu - self.threshold_value, 0) * cp.sign(coeffs_gpu)
        elif self.threshold_method == 'hard':
            return coeffs_gpu * (cp.abs(coeffs_gpu) >= self.threshold_value)
        else:
            raise ValueError("Invalid thresholding method. Choose 'soft' or 'hard'.")
