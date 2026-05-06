import numpy as np
from scipy.signal import detrend, butter, filtfilt

#Detrending
def remove_trend(signal):
    signal=np.array(signal)
    if len(signal) < 10:
        return signal
    return detrend(signal)

#Bandpass filtering
def bandpass_filter(signal,fs=30, low=0.7,high=4):
    signal=np.array(signal)
    if len(signal) < 30:
        return signal
    nyquist=0.5 * fs
    low=low/nyquist
    high=high/nyquist
    b,a = butter(3,[low,high],btype='band')
    return filtfilt(b,a,signal)
#Proceessing pipeline
def process_signal(signal):
    signal=remove_trend(signal)
    signal=bandpass_filter(signal)
    return signal

