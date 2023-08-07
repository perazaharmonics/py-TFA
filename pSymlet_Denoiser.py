import numpy as np
import pywt

class SymletWaveletDenoiser:
    def __init__(self, wavelet='sym2', threshold_method='soft', threshold_value=0.1):
        self.wavelet = wavelet
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def denoise(self, signal):
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet)
        thresholded_coeffs = [self._threshold(c) for c in coeffs]
        denoised_signal = pywt.waverec(thresholded_coeffs, wavelet=self.wavelet)
        return denoised_signal

    def _threshold(self, coeffs):
        if self.threshold_method == 'soft':
            return pywt.threshold(coeffs, self.threshold_value, mode='soft')
        elif self.threshold_method == 'hard':
            return pywt.threshold(coeffs, self.threshold_value, mode='hard')
        else:
            raise ValueError("Invalid thresholding method. Choose 'soft' or 'hard'.")
