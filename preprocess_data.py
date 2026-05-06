"""
preprocess_data_mediapipe.py

Improvements over preprocess_data.py:
  1. Subject-name mapped output:  subject_001.npz → subject_001.npz  (not sample_0.npz)
  2. MediaPipe face detection     (replaces unreliable Haar cascade)
  3. Forehead + cheek ROI         (richer pulse signal than full face crop)
  4. Same pipeline structure      (drop-in replacement — train.py unchanged)

Run this in a SEPARATE Colab tab while training runs in another.
Install first:  pip install mediapipe
"""

import os
import json
import numpy as np
import cv2
from scipy.signal import detrend, butter, filtfilt

# ── Block TensorFlow before MediaPipe loads (fixes Colab protobuf conflict) ──
import sys
import types

# Block TensorFlow/protobuf conflict with MediaPipe
_doc = types.ModuleType('tensorflow.tools.docs.doc_controls')
_doc.do_not_generate_docs = lambda f: f  # mock the decorator
_doc.do_not_build_docs    = lambda f: f
sys.modules['tensorflow']                          = types.ModuleType('tensorflow')
sys.modules['tensorflow.tools']                    = types.ModuleType('tensorflow.tools')
sys.modules['tensorflow.tools.docs']               = types.ModuleType('tensorflow.tools.docs')
sys.modules['tensorflow.tools.docs.doc_controls']  = _doc

# ── MediaPipe ─────────────────────────────────────────────────────────
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import face_detection as mp_face_detection


# ─────────────────────────────────────────
# Signal helpers  (unchanged from original)
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
# ROI extraction helpers
# ─────────────────────────────────────────

# MediaPipe Face Mesh landmark indices
# Forehead: landmarks around top of face
# Cheeks: left and right cheek regions
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                      361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                      176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                      162, 21, 54, 103, 67, 109]

LEFT_CHEEK_LANDMARKS  = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
RIGHT_CHEEK_LANDMARKS = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]


def extract_roi_mediapipe(frame_bgr, face_mesh, target_size=(72, 72)):
    """
    Extract forehead + cheek ROI using MediaPipe Face Mesh.
    Uses weighted average: forehead=0.5, left cheek=0.25, right cheek=0.25
    Forehead weighted higher — denser superficial vasculature, less motion artifact.
    Falls back to full-frame resize if face not detected.

    Returns: resized ROI (target_size, target_size, 3) float32 in [0,1]
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        # fallback: return full frame resized
        frame_out = cv2.resize(frame_bgr, target_size)
        return frame_out.astype(np.float32) / 255.0

    lm = results.multi_face_landmarks[0].landmark

    # ── forehead box ──────────────────────────────────────────────────
    # Use landmarks 10 (top center) and surrounding to get forehead region
    nose_tip   = lm[1]
    forehead_c = lm[10]

    fx = int(forehead_c.x * w)
    fy = int(forehead_c.y * h)
    nx = int(nose_tip.x * w)
    ny = int(nose_tip.y * h)

    face_h = abs(ny - fy) * 2  # rough face height estimate
    face_w = face_h

    # Forehead: top 25% of face
    fh_x1 = max(0, fx - face_w // 4)
    fh_x2 = min(w, fx + face_w // 4)
    fh_y1 = max(0, fy - face_h // 10)
    fh_y2 = min(h, fy + face_h // 6)

    # ── cheek boxes ───────────────────────────────────────────────────
    def landmark_box(indices, pad=10):
        xs = [int(lm[i].x * w) for i in indices]
        ys = [int(lm[i].y * h) for i in indices]
        return (max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(w, max(xs) + pad), min(h, max(ys) + pad))

    lc = landmark_box(LEFT_CHEEK_LANDMARKS)
    rc = landmark_box(RIGHT_CHEEK_LANDMARKS)

    # ── crop each region and resize to target ─────────────────────────
    def safe_crop(img, box):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, target_size).astype(np.float32) / 255.0

    forehead_roi = safe_crop(frame_bgr, (fh_x1, fh_y1, fh_x2, fh_y2))
    left_cheek   = safe_crop(frame_bgr, lc)
    right_cheek  = safe_crop(frame_bgr, rc)

    # Weighted average: forehead=0.5, left cheek=0.25, right cheek=0.25
    # Forehead has strongest pulse signal and least motion artifact
    roi_weight_pairs = [
        (forehead_roi, 0.5),
        (left_cheek,   0.25),
        (right_cheek,  0.25),
    ]

    valid_pairs = [(r, w) for r, w in roi_weight_pairs if r is not None]

    if not valid_pairs:
        return cv2.resize(frame_bgr, target_size).astype(np.float32) / 255.0

    # Renormalize weights if some ROIs failed detection
    total_weight = sum(w for _, w in valid_pairs)
    result = sum(r * (w / total_weight) for r, w in valid_pairs)

    return np.clip(result, 0, 1).astype(np.float32)


# ─────────────────────────────────────────
# Main extraction class
# ─────────────────────────────────────────

class RPPGDatasetMediaPipe:
    def __init__(self, root_dir, window_size=128, stride=32):
        self.root_dir    = root_dir
        self.window_size = window_size
        self.stride      = stride
        self.samples     = self._load_samples()
    
    def __len__(self):                   
        return len(self.samples)

    def _load_samples(self):
        """
        Returns list of (subject_name, video_path, json_path)
        subject_name is the folder name → used for .npz filename mapping
        """
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
                samples.append((subject, video, json_f))  # ← subject name included

        print(f"Found {len(samples)} subjects: {[s[0] for s in samples[:5]]}...")
        return samples

    def extract_frames(self, video_path, target_size=(72, 72)):
        """Extract frames using MediaPipe face mesh + ROI averaging."""
        cap    = cv2.VideoCapture(video_path)
        frames = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                roi = extract_roi_mediapipe(frame, face_mesh, target_size)
                frames.append(roi)

        cap.release()
        return np.array(frames)  # (T, H, W, 3)

    def extract_motion(self, frames):
        return (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + 1e-6)

    def extract_gt_signal(self, json_path, frames):
        """Unchanged from original — same JSON parsing."""
        with open(json_path) as f:
            data = json.load(f)

        for scenario in data.get("scenarios", []):
            rec = scenario.get("recordings", {})

            if "ppg" in rec:
                try:
                    ppg_ts  = rec["ppg"]["timeseries"]
                    ppg_sig = np.array([x[1] for x in ppg_ts], dtype=np.float32)
                    ppg_t   = np.array([x[0] for x in ppg_ts], dtype=np.float32)

                    ppg_t    = (ppg_t - ppg_t[0]) / 1000.0
                    duration = ppg_t[-1]
                    if duration <= 0:
                        continue

                    fs_ppg = len(ppg_t) / duration
                    signal = _process(ppg_sig, fs=fs_ppg)
                    signal = _normalise(signal)

                    n_frames = len(frames)
                    signal = np.interp(
                        np.linspace(0, len(signal) - 1, n_frames),
                        np.arange(len(signal)),
                        signal
                    ).astype(np.float32)

                    signal = signal[:n_frames - 1]
                    print("  Using REAL PPG")
                    return signal

                except Exception as e:
                    print(f"  PPG error: {e}")
                    continue

        print("  No valid PPG → skipping")
        return None

    def create_windows(self, frames, motion, signal):
        ws, stride = self.window_size, self.stride
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


# ─────────────────────────────────────────
# Main preprocessing script
# ─────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "/content/drive/MyDrive/shared/FYP/data/data/videos"
    SAVE_PATH = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"  # ← separate folder!
    os.makedirs(SAVE_PATH, exist_ok=True)

    dataset = RPPGDatasetMediaPipe(DATA_PATH)
    total   = len(dataset)
    print(f"Total subjects: {total}\n")

    done = skipped = failed = 0

    for subject_name, video_path, json_path in dataset.samples:
        # ── SUBJECT-MAPPED filename ─────────────────────────────────
        save_file = os.path.join(SAVE_PATH, f"{subject_name}.npz")

        if os.path.exists(save_file):
            skipped += 1
            print(f"[SKIP] {subject_name}")
            continue

        print(f"[PROCESS] {subject_name}")
        try:
            frames = dataset.extract_frames(video_path)
            motion = dataset.extract_motion(frames)
            signal = dataset.extract_gt_signal(json_path, frames)

            if signal is None:
                print(f"  ERROR: no signal")
                failed += 1
                continue

            apps, mots, sigs = dataset.create_windows(frames, motion, signal)

            if len(apps) == 0:
                print(f"  ERROR: no valid windows")
                failed += 1
                continue

            app_arr = np.array(apps, dtype=np.float32)
            mot_arr = np.array(mots, dtype=np.float32)
            sig_arr = np.array(sigs, dtype=np.float32)

            np.savez_compressed(
                save_file,
                appearance=app_arr,
                motion=mot_arr,
                signal=sig_arr,
                subject=subject_name   # ← also store name inside npz
            )
            print(f"  Saved → {subject_name}.npz | windows={len(apps)} | sig_std={sig_arr.std():.3f}")
            done += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\nDone={done}  Skipped={skipped}  Failed={failed}")
    print(f"Saved to: {SAVE_PATH}")