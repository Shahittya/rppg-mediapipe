import numpy as np

def calculate_heart_rate(signal, fps, prev_bpm=None):

    signal = np.array(signal)

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    n = len(signal)

    freqs = np.fft.rfftfreq(n, d=1/fps)
    fft_values = np.abs(np.fft.rfft(signal))

    # Focus on realistic HR range
    mask = (freqs >= 0.8) & (freqs <= 2.0)
    freqs = freqs[mask]
    fft_values = fft_values[mask]

    # Sort peaks by strength
    sorted_indices = np.argsort(fft_values)[::-1]
    candidate_bpms = freqs[sorted_indices] * 60

    # Keep realistic HR values
    candidate_bpms = [b for b in candidate_bpms if 50 <= b <= 110]

    if not candidate_bpms:
        return 0, freqs, fft_values

    # SMART SELECTION (NO LOCK-IN)
    if prev_bpm is not None:
        close_candidates = [b for b in candidate_bpms if abs(b - prev_bpm) < 20]

        if close_candidates:
            bpm = close_candidates[0]
        else:
            bpm = candidate_bpms[0]
    else:
        bpm = candidate_bpms[0]

    return bpm, freqs, fft_values   # 🔥 VERY IMPORTANT