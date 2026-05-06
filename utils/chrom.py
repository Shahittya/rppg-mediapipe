import numpy as np
from scipy.signal import detrend, butter, filtfilt


def _bandpass(signal, fs=30, low=0.7, high=4.0):
    signal = np.array(signal, dtype=np.float32)
    if len(signal) < 30:
        return signal
    nyq  = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal).astype(np.float32)


def chrom_method(r, g, b, fs=30):
    """
    FIX: added detrend + bandpass after computing the CHROM signal.

    Old code returned the raw CHROM output with no filtering.
    Low-frequency skin colour drift caused the raw CHROM signal to have
    very high SNR (strong DC + slow drift dominates full-spectrum power),
    so select_best() always picked CHROM even when it was noisier in the
    actual heart rate band.  Filtering removes that drift so SNR comparison
    is fair.
    """
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    X = 3 * r - 2 * g
    Y = 1.5 * r + g - 1.5 * b

    std2 = np.std(Y)
    if std2 == 0:
        return np.zeros_like(r)

    alpha = np.std(X) / std2
    s     = X - alpha * Y

    # Detrend
    if len(s) >= 10:
        s = detrend(s)

    # Bandpass to HR band only
    s = _bandpass(s, fs=fs)

    return s.astype(np.float32)
