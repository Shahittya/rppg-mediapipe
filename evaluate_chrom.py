"""
evaluate_chrom.py  — Correct version

Reads raw video files directly (not .npz)
GT BPM comes from JSON files (real sensor ground truth)
CHROM runs on actual raw RGB from face ROI

Place in: rppg-mediapipe-main/
Run:      !python evaluate_chrom.py

Requires:  pip install mediapipe==0.10.13
"""

import os
import sys
import types
import json
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from torch.utils.data import random_split

# ── Block TF conflict before MediaPipe ────────────────────────────────
_doc = types.ModuleType('tensorflow.tools.docs.doc_controls')
_doc.do_not_generate_docs = lambda f: f
_doc.do_not_build_docs    = lambda f: f
sys.modules['tensorflow']                         = types.ModuleType('tensorflow')
sys.modules['tensorflow.tools']                   = types.ModuleType('tensorflow.tools')
sys.modules['tensorflow.tools.docs']              = types.ModuleType('tensorflow.tools.docs')
sys.modules['tensorflow.tools.docs.doc_controls'] = _doc

from mediapipe.python.solutions import face_mesh as mp_face_mesh

from dataset_fast import RPPGFastDataset
from utils.chrom import chrom_method
from utils.heart_rate import calculate_heart_rate

# ── Config ────────────────────────────────────────────────────────────
VIDEO_PATH = "/content/drive/MyDrive/shared/FYP/data/data/videos"
NPZ_PATH   = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"
PLOT_PATH  = "/content/drive/MyDrive/shared/FYP/eval_plots"
FPS        = 30
SEED       = 42

os.makedirs(PLOT_PATH, exist_ok=True)

# ── Same val split as training ────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

dataset    = RPPGFastDataset(NPZ_PATH)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

_, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Get the actual file paths for val subjects
val_files   = [dataset.files[i] for i in val_ds.indices]
val_subjects = [os.path.splitext(os.path.basename(f))[0] for f in val_files]
print(f"Val subjects ({len(val_subjects)}): {val_subjects[:5]}...")


# ── Signal helpers ────────────────────────────────────────────────────

def bandpass(signal, fs, low=0.7, high=3.0):
    if len(signal) < 30:
        return signal
    nyq = 0.5 * fs
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


def pearson_r(pred, gt):
    p = pred - pred.mean()
    g = gt   - gt.mean()
    denom = np.sqrt((p**2).sum() * (g**2).sum()) + 1e-8
    return float((p * g).sum() / denom)


def get_bpm(signal, prev_bpm=None):
    bpm, _, _ = calculate_heart_rate(signal, FPS, prev_bpm)
    return bpm if bpm > 0 else None


# ── Load GT BPM from JSON ─────────────────────────────────────────────

def load_gt_from_json(json_path):
    """Returns (gt_bpm, gt_signal, fs_ppg) from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    for scenario in data.get("scenarios", []):
        rec = scenario.get("recordings", {})
        if "ppg" not in rec:
            continue
        try:
            ppg_ts  = rec["ppg"]["timeseries"]
            ppg_sig = np.array([x[1] for x in ppg_ts], dtype=np.float32)
            ppg_t   = np.array([x[0] for x in ppg_ts], dtype=np.float32)
            ppg_t   = (ppg_t - ppg_t[0]) / 1000.0
            duration = ppg_t[-1]
            if duration <= 0:
                continue

            fs_ppg  = len(ppg_t) / duration
            sig     = detrend(ppg_sig)
            sig     = bandpass(sig, fs=fs_ppg)
            sig_norm = (sig - sig.mean()) / (sig.std() + 1e-6)

            # BPM from GT signal
            gt_bpm, _, _ = calculate_heart_rate(sig_norm, fs_ppg)
            if gt_bpm > 0:
                return gt_bpm, sig_norm, fs_ppg
        except Exception as e:
            print(f"  JSON error: {e}")
            continue
    return None, None, None


# ── Extract RGB from video using MediaPipe face ROI ───────────────────

def extract_rgb_from_video(video_path, target_frames=None):
    """
    Reads video, detects face with MediaPipe, extracts upper-face ROI,
    returns mean R, G, B per frame as numpy arrays.
    """
    cap    = cv2.VideoCapture(video_path)
    r_list, g_list, b_list = [], [], []

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

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Upper face crop (same as preprocess_data.py — consistent ROI)
                forehead_lm = lm[10]
                chin_lm     = lm[152]
                left_lm     = lm[234]
                right_lm    = lm[454]

                fy = int(forehead_lm.y * h)
                cy = int(chin_lm.y * h)
                lx = int(left_lm.x * w)
                rx = int(right_lm.x * w)

                face_h = cy - fy
                pad_x  = int((rx - lx) * 0.08)

                x1 = max(0, lx - pad_x)
                x2 = min(w, rx + pad_x)
                y1 = max(0, fy)
                y2 = max(0, fy + int(face_h * 0.75))

                if y2 > y1 and x2 > x1:
                    roi = frame[y1:y2, x1:x2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
                    r_list.append(roi_rgb[:, :, 0].mean())
                    g_list.append(roi_rgb[:, :, 1].mean())
                    b_list.append(roi_rgb[:, :, 2].mean())
                    continue

            # Fallback: full frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            r_list.append(frame_rgb[:, :, 0].mean())
            g_list.append(frame_rgb[:, :, 1].mean())
            b_list.append(frame_rgb[:, :, 2].mean())

    cap.release()
    return np.array(r_list), np.array(g_list), np.array(b_list)


# ── Run CHROM on each val subject ─────────────────────────────────────

all_chrom_bpm = []
all_gt_bpm    = []
all_pearson   = []
subject_rows  = []
waveform_samples = []

for subj_name in val_subjects:
    subj_dir = os.path.join(VIDEO_PATH, subj_name)

    # Find video and json
    video_file = json_file = None
    if not os.path.isdir(subj_dir):
        print(f"[SKIP] {subj_name} — folder not found")
        continue

    for f in os.listdir(subj_dir):
        if f.endswith('_1.mp4'):
            video_file = os.path.join(subj_dir, f)
        elif f.endswith('.json'):
            json_file = os.path.join(subj_dir, f)

    if not video_file or not json_file:
        print(f"[SKIP] {subj_name} — missing video or json")
        continue

    print(f"[PROCESS] {subj_name}")

    # ── GT from JSON ──────────────────────────────────────────────────
    gt_bpm, gt_sig, fs_ppg = load_gt_from_json(json_file)
    if gt_bpm is None:
        print(f"  No GT signal")
        continue

    # ── CHROM from raw video ──────────────────────────────────────────
    try:
        r, g, b = extract_rgb_from_video(video_file)
    except Exception as e:
        print(f"  Video error: {e}")
        continue

    if len(r) < 64:
        print(f"  Too few frames: {len(r)}")
        continue

    # Run CHROM on full video signal
    chrom_sig = chrom_method(r, g, b, fs=FPS)

    # BPM from CHROM signal
    c_bpm = get_bpm(chrom_sig)
    if c_bpm is None:
        print(f"  CHROM BPM failed")
        continue

    mae_subj = abs(c_bpm - gt_bpm)
    r_subj   = pearson_r(
        np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(chrom_sig)), chrom_sig),
        np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(gt_sig)),   gt_sig)
    )

    print(f"  GT={gt_bpm:.1f}  CHROM={c_bpm:.1f}  MAE={mae_subj:.2f}  r={r_subj:.3f}")

    subject_rows.append({
        "name":      subj_name,
        "gt_bpm":    gt_bpm,
        "chrom_bpm": c_bpm,
        "mae":       mae_subj,
        "pearson":   r_subj,
    })

    all_chrom_bpm.append(c_bpm)
    all_gt_bpm.append(gt_bpm)
    all_pearson.append(r_subj)

    # Save waveform sample (first 4)
    if len(waveform_samples) < 4:
        t_len = min(len(chrom_sig), len(gt_sig), 128 * 10)
        cs = chrom_sig[:t_len]
        gs = gt_sig[:t_len] if len(gt_sig) >= t_len else np.interp(
            np.linspace(0, 1, t_len), np.linspace(0, 1, len(gt_sig)), gt_sig)
        waveform_samples.append({
            "name":  subj_name,
            "chrom": cs,
            "gt":    gs,
            "r":     r_subj,
            "c_bpm": c_bpm,
            "g_bpm": gt_bpm,
        })


# ── Overall metrics ───────────────────────────────────────────────────
if not all_chrom_bpm:
    print("No subjects evaluated — check VIDEO_PATH")
else:
    all_c = np.array(all_chrom_bpm)
    all_g = np.array(all_gt_bpm)

    mae_overall  = np.mean(np.abs(all_c - all_g))
    rmse_overall = np.sqrt(np.mean((all_c - all_g)**2))
    r_overall    = np.mean(all_pearson)

    print("\n" + "=" * 50)
    print("   CHROM EVALUATION — Val Set (vs JSON GT)")
    print("=" * 50)
    print(f"  MAE      : {mae_overall:.2f} BPM")
    print(f"  RMSE     : {rmse_overall:.2f} BPM")
    print(f"  Pearson  : {r_overall:.4f}")
    print(f"  Subjects : {len(subject_rows)}")
    print("=" * 50)

    print(f"\n{'Subject':<12}  {'GT BPM':>8}  {'CHROM':>8}  {'MAE':>6}  {'Pearson':>8}")
    print("-" * 52)
    for r in sorted(subject_rows, key=lambda x: x["mae"]):
        print(f"  {r['name']:<10}  {r['gt_bpm']:>7.2f}  {r['chrom_bpm']:>7.2f}  "
              f"{r['mae']:>5.2f}   {r['pearson']:>7.4f}")

    print(f"\n--- For your thesis comparison table ---")
    print(f"  CHROM MAE (this script):     {mae_overall:.2f} BPM  ← from raw video + JSON GT")
    print(f"  DeepPhysLSTM MAE:              5.62 BPM  ← from evaluate.py")
    print(f"  Community CHROM benchmark:    ~7.0 BPM")

    # ── Plot 1: Waveform comparison ───────────────────────────────────
    if waveform_samples:
        n = len(waveform_samples)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
        if n == 1:
            axes = [axes]

        for ax, sample in zip(axes, waveform_samples):
            def norm(s):
                return (s - s.mean()) / (s.std() + 1e-8)

            t = np.arange(len(sample["chrom"])) / FPS
            t_gt = np.arange(len(sample["gt"])) / fs_ppg if fs_ppg else t

            ax.plot(t_gt[:len(sample["gt"])], norm(sample["gt"]),
                    color="#4CAF50", linewidth=1.5,
                    label=f"GT PPG ({sample['g_bpm']:.1f} BPM)", alpha=0.9)
            ax.plot(t[:len(sample["chrom"])], norm(sample["chrom"]),
                    color="#FF9800", linewidth=1.5,
                    label=f"CHROM ({sample['c_bpm']:.1f} BPM)",
                    linestyle="--", alpha=0.9)
            ax.set_title(f"{sample['name']}  |  Pearson r = {sample['r']:.3f}", fontsize=11)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalised amplitude")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle("CHROM vs Ground Truth PPG (from JSON sensor)",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        p1 = os.path.join(PLOT_PATH, "chrom_vs_gt_waveform.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"\nSaved: {p1}")

    # ── Plot 2: BPM scatter ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(all_gt_bpm, all_chrom_bpm, alpha=0.7,
               color="#FF9800", s=80, label="Subjects", zorder=3)

    mn = min(min(all_gt_bpm), min(all_chrom_bpm)) - 5
    mx = max(max(all_gt_bpm), max(all_chrom_bpm)) + 5
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect")
    ax.fill_between([mn, mx], [mn-5, mx-5], [mn+5, mx+5],
                    alpha=0.1, color="green", label="±5 BPM")

    # Label each point
    for r in subject_rows:
        ax.annotate(r["name"], (r["gt_bpm"], r["chrom_bpm"]),
                    fontsize=7, alpha=0.7, xytext=(3, 3),
                    textcoords="offset points")

    ax.set_xlabel("Ground Truth BPM (JSON sensor)", fontsize=12)
    ax.set_ylabel("CHROM Predicted BPM", fontsize=12)
    ax.set_title(f"CHROM — MAE={mae_overall:.2f}  RMSE={rmse_overall:.2f}  r={r_overall:.3f}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    p2 = os.path.join(PLOT_PATH, "chrom_bpm_scatter.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {p2}")

    # ── Plot 3: Per-subject MAE ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    names  = [r["name"]  for r in subject_rows]
    maes   = [r["mae"]   for r in subject_rows]
    colors = ["#4CAF50" if m < 5 else "#FF9800" if m < 10 else "#F44336" for m in maes]

    ax.bar(range(len(names)), maes, color=colors, edgecolor="white")
    ax.axhline(mae_overall, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean MAE = {mae_overall:.2f} BPM")
    ax.axhline(5, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
               label="5 BPM threshold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (BPM)", fontsize=12)
    ax.set_title("CHROM Per-Subject MAE vs JSON Ground Truth  |  Green<5  Orange<10  Red≥10",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    p3 = os.path.join(PLOT_PATH, "chrom_per_subject_mae.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {p3}")

    print(f"\nAll plots saved to: {PLOT_PATH}")
    print("Done.")
