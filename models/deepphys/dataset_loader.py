import os
import json
import numpy as np
import cv2
from scipy.signal import detrend, butter, filtfilt


# ─────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────

def _bandpass(signal, fs, low=0.7, high=3.0):
    if len(signal) < 30:
        return signal
    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def _process(signal, fs):
    if len(signal) > 10:
        signal = detrend(signal)
    return _bandpass(signal, fs)


def _normalise(signal):
    std = np.std(signal)
    return (signal - np.mean(signal)) / std if std > 1e-6 else signal


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class RPPGDataset:
    def __init__(self, root_dir, window_size=128, stride=32):
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride

        self.samples = self._load_samples()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _load_samples(self):
        samples = []
        for subject in sorted(os.listdir(self.root_dir)):
            sp = os.path.join(self.root_dir, subject)
            if not os.path.isdir(sp):
                continue

            video = json_f = None
            for f in os.listdir(sp):
                if f.endswith('_1.mp4'):
                    video = os.path.join(sp, f)
                elif f.endswith('.json'):
                    json_f = os.path.join(sp, f)

            if video and json_f:
                samples.append((video, json_f))

        return samples

    # ─────────────────────────────
    # Frame extraction
    # ─────────────────────────────

    def extract_frames(self, video_path, target_size=(72, 72)):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                frame = frame[y:y+h, x:x+w]

            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()
        return np.array(frames)

    # ─────────────────────────────
    # Motion
    # ─────────────────────────────

    def extract_motion(self, frames):
        return (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + 1e-6)

    # ─────────────────────────────
    # Ground truth (FIXED)
    # ─────────────────────────────

    def extract_gt_signal(self, json_path, frames):
        with open(json_path) as f:
            data = json.load(f)

        for scenario in data.get("scenarios", []):
            rec = scenario.get("recordings", {})

            if "ppg" in rec:
                try:
                    ppg_ts = rec["ppg"]["timeseries"]

                    ppg_sig = np.array([x[1] for x in ppg_ts], dtype=np.float32)
                    ppg_t   = np.array([x[0] for x in ppg_ts], dtype=np.float32)

                    # ── FIX 1: correct timestamps ──
                    ppg_t = (ppg_t - ppg_t[0]) / 1000.0
                    duration = ppg_t[-1]

                    if duration <= 0:
                        continue

                    fs_ppg = len(ppg_t) / duration

                    # ── FIX 2: correct processing ──
                    signal = _process(ppg_sig, fs=fs_ppg)
                    signal = _normalise(signal)

                    # ── FIX 3: match frames (NO compression bug) ──
                    n_frames = len(frames)

                    signal = np.interp(
                        np.linspace(0, len(signal)-1, n_frames),
                        np.arange(len(signal)),
                        signal
                    ).astype(np.float32)

                    # match motion length
                    signal = signal[:n_frames-1]

                    print("Using REAL PPG")
                    return signal

                except Exception as e:
                    print("PPG error:", e)
                    continue

        print("No valid PPG → skipping")
        return None

    # ─────────────────────────────
    # Windowing
    # ─────────────────────────────

    def create_windows(self, frames, motion, signal):
        ws = self.window_size
        stride = self.stride

        T = len(motion)
        if T < ws:
            return [], [], []

        apps, mots, sigs = [], [], []

        for start in range(0, T - ws + 1, stride):
            end = start + ws

            a = frames[start:end]
            m = motion[start:end]
            s = signal[start:end]

            if np.std(s) < 0.01:
                continue

            apps.append(a)
            mots.append(m)
            sigs.append(s)

        return apps, mots, sigs

    # ─────────────────────────────
    # Dataset interface
    # ─────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, json_path = self.samples[idx]

        frames = self.extract_frames(video_path)
        motion = self.extract_motion(frames)

        signal = self.extract_gt_signal(json_path, frames)

        if signal is None:
            raise ValueError("Bad signal")

        apps, mots, sigs = self.create_windows(frames, motion, signal)

        if len(apps) == 0:
            raise ValueError("No valid windows")

        return np.array(apps), np.array(mots), np.array(sigs)