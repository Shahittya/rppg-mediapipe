import numpy as np
from scipy.signal import butter, filtfilt


def normalize(x):
    x = np.array(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def _bandpass_power(signal, fs=30, low=0.7, high=2.5):
    """
    Compute signal power ONLY inside the HR band (0.7-2.5 Hz).
    This stops low-freq skin drift or high-freq noise from inflating SNR.
    Previously compute_snr() used the full FFT spectrum — that let a
    signal dominated by drift score higher than a clean pulse signal.
    """
    signal = np.array(signal, dtype=np.float32)
    n      = len(signal)
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    power  = np.abs(np.fft.rfft(signal)) ** 2

    hr_mask    = (freqs >= low)  & (freqs <= high)
    noise_mask = ~hr_mask

    hr_power    = np.sum(power[hr_mask])
    noise_power = np.sum(power[noise_mask]) + 1e-6

    return hr_power / noise_power   # true in-band SNR


def compute_snr(signal, fs=30):
    """Public alias — used by inference.py and other callers."""
    return _bandpass_power(signal, fs=fs)


def simple_fusion(chrom, deep):
    return 0.5 * np.array(chrom) + 0.5 * np.array(deep)


def weighted_fusion(chrom, deep, fs=30):
    snr_c = _bandpass_power(chrom, fs=fs)
    snr_d = _bandpass_power(deep,  fs=fs)
    w_c   = snr_c / (snr_c + snr_d + 1e-6)
    w_d   = 1.0 - w_c
    return w_c * np.array(chrom) + w_d * np.array(deep)


def select_best(chrom, deep, fs=30):
    """
    Return whichever signal has higher in-band SNR (0.7-2.5 Hz).
    Previously used full-spectrum SNR which always favoured the noisier
    signal if it had strong low-frequency drift.
    """
    snr_c = _bandpass_power(chrom, fs=fs)
    snr_d = _bandpass_power(deep,  fs=fs)
    return chrom if snr_c > snr_d else deep
