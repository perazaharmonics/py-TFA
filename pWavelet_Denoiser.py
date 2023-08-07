import pywt
import numpy as np

class WaveletDenoiser:
    def __init__(self, wavelet='db1', level=None, threshold_method='soft', threshold_value=0.1):
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value

    def denoise(self, signal):
        # Check if the signal is stereo (2D) or mono (1D)
        if len(signal.shape) == 1:
            return self._wavelet_denoise(signal)
        elif len(signal.shape) == 2:
            # Handle stereo signal by denoising both channels
            channels = [self._wavelet_denoise(channel) for channel in signal.T]
            return np.vstack(channels).T
        else:
            raise ValueError("Input data must be 1-dimensional (mono) or 2-dimensional (stereo).")

    def _wavelet_denoise(self, signal):
        # Decompose the signal using wavelet transform
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=self.level)

        # Threshold the detail coefficients
        thresholded_coeffs = [coeffs[0]]  # Keep the approximation coefficients
        for detail_coeff in coeffs[1:]:
            if self.threshold_method == 'soft':
                thresholded_coeffs.append(pywt.threshold(detail_coeff, self.threshold_value, mode='soft'))
            elif self.threshold_method == 'hard':
                thresholded_coeffs.append(pywt.threshold(detail_coeff, self.threshold_value, mode='hard'))
            else:
                raise ValueError("Invalid thresholding method. Choose 'soft' or 'hard'.")

        # Reconstruct the denoised signal
        denoised_signal = pywt.waverec(thresholded_coeffs, wavelet=self.wavelet)

        return denoised_signal
