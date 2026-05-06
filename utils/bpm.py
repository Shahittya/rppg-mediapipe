import numpy as np

def get_bpm(signal, fps=30):
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fps)

    mask = (freqs >= 0.7) & (freqs <= 4.0)

    if np.sum(mask) == 0:
        return 0

    peak = freqs[mask][np.argmax(np.abs(fft[mask]))]
    return peak * 60