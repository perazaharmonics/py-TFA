import pywt
import numpy as np

class ContinuousWaveletDenoiser:
    def __init__(self, wavelet='cmor', scales=None, threshold_method='soft', threshold_value=0.1):
        if wavelet not in pywt.wavelist(kind='continuous'):
            raise ValueError("Invalid wavelet. Choose a valid continuous wavelet from the PyWavelets library.")

        self.wavelet = wavelet
        self.scales = scales if scales else np.arange(1, 128)
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def denoise(self, input_signal):
        # Compute the Continuous Wavelet Transform (CWT)
        coeffs, _ = pywt.cwt(input_signal, scales=self.scales, wavelet=self.wavelet)

        # Threshold the wavelet coefficients
        if self.threshold_method == 'soft':
            thresholded_coeffs = pywt.threshold(coeffs, self.threshold_value, mode='soft')
        elif self.threshold_method == 'hard':
            thresholded_coeffs = pywt.threshold(coeffs, self.threshold_value, mode='hard')
        else:
            raise ValueError("Invalid thresholding method. Choose 'soft' or 'hard'.")

        # Reconstruct the denoised signal
        denoised_signal = self._inverse_cwt(thresholded_coeffs)

        return denoised_signal

    def _inverse_cwt(self, coeffs):
        # Sum across scales to reconstruct the signal
        # Weighting by the scale spacing can be considered here, depending on the chosen wavelet and scales
        return np.sum(coeffs, axis=0)
