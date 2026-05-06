import numpy as np

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def compute_snr(signal):
    fft = np.fft.fft(signal)
    power = np.abs(fft) ** 2
    return np.max(power) / (np.mean(power) + 1e-6)

def simple_fusion(chrom, deep):
    return 0.5 * chrom + 0.5 * deep

def weighted_fusion(chrom, deep):
    snr_c = compute_snr(chrom)
    snr_d = compute_snr(deep)

    w_c = snr_c / (snr_c + snr_d + 1e-6)
    w_d = 1 - w_c

    return w_c * chrom + w_d * deep

def select_best(chrom, deep):
    snr_c = compute_snr(chrom)
    snr_d = compute_snr(deep)

    return chrom if snr_c > snr_d else deep