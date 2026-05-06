import numpy as np

def calculate_heart_rate(signal, fps, prev_bpm=None):

    signal = np.array(signal)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    n = len(signal)
    freqs      = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_values = np.abs(np.fft.rfft(signal))

    # FIX: use 0.7-2.5 Hz (42-150 BPM)
    # Old range 0.8-2.0 Hz (48-120 BPM) was cutting off valid peaks
    mask = (freqs >= 0.7) & (freqs <= 2.5)
    freqs_masked = freqs[mask]
    fft_masked   = fft_values[mask]

    sorted_indices  = np.argsort(fft_masked)[::-1]
    candidate_bpms  = freqs_masked[sorted_indices] * 60

    candidate_bpms = [b for b in candidate_bpms if 42 <= b <= 150]

    if not candidate_bpms:
        return 0, freqs_masked, fft_masked

    if prev_bpm is not None:
        close = [b for b in candidate_bpms if abs(b - prev_bpm) < 20]
        bpm   = close[0] if close else candidate_bpms[0]
    else:
        bpm = candidate_bpms[0]

    return bpm, freqs_masked, fft_masked
