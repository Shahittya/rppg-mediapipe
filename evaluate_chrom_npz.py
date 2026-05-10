"""
evaluate_chrom.py

Runs CHROM on the same val subjects as your DL model.
Uses identical split (seed=42) for fair comparison.

Place in: rppg-mediapipe-main/  (same folder as train.py)
Run:      !python evaluate_chrom.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset_fast import RPPGFastDataset
from utils.chrom import chrom_method
from utils.heart_rate import calculate_heart_rate

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"
PLOT_PATH = "/content/drive/MyDrive/shared/FYP/eval_plots"
FPS       = 30
SEED      = 42

os.makedirs(PLOT_PATH, exist_ok=True)

# ── Same split as training ────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────

def pearson_r(pred, gt):
    p = pred - pred.mean()
    g = gt   - gt.mean()
    denom = np.sqrt((p**2).sum() * (g**2).sum()) + 1e-8
    return float((p * g).sum() / denom)


def get_bpm(signal, prev_bpm=None):
    bpm, _, _ = calculate_heart_rate(signal, FPS, prev_bpm)
    return bpm if bpm > 0 else None


def chrom_from_appearance(appearance_window):
    """
    appearance_window: (T, H, W, 3) float32 in [-1, 1]
    Reverses dataset_fast normalization → spatial average → CHROM
    """
    # reverse (x-0.5)/0.5 → back to [0,1]
    app = (appearance_window * 0.5) + 0.5
    app = np.clip(app, 0, 1)

    # spatial average per frame → RGB timeseries
    r = app[:, :, :, 0].mean(axis=(1, 2))
    g = app[:, :, :, 1].mean(axis=(1, 2))
    b = app[:, :, :, 2].mean(axis=(1, 2))

    return chrom_method(r, g, b, fs=FPS)


# ── Load val set ──────────────────────────────────────────────────────
dataset    = RPPGFastDataset(DATA_PATH)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

_, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
print(f"Val subjects: {len(val_ds)}\n")


# ── Run CHROM on val set ──────────────────────────────────────────────
all_pred_bpm = []
all_gt_bpm   = []
all_pearson  = []
subject_rows = []
waveform_samples = []

for subj_idx, (appearance, motion, signal) in enumerate(val_loader):
    appearance_np = appearance.squeeze(0).numpy()  # (N, T, H, W, 3)
    signal_np     = signal.squeeze(0).numpy()      # (N, T)

    subj_pred_bpms = []
    subj_gt_bpms   = []
    subj_pearsons  = []
    prev_bpm       = None

    for i in range(len(appearance_np)):
        s     = signal_np[i]
        a_np  = appearance_np[i]          # (T, H, W, 3) in [-1,1]

        chrom_sig = chrom_from_appearance(a_np)

        c_bpm = get_bpm(chrom_sig, prev_bpm)
        g_bpm = get_bpm(s)

        if not (c_bpm and g_bpm):
            continue

        prev_bpm = c_bpm

        subj_pred_bpms.append(c_bpm)
        subj_gt_bpms.append(g_bpm)
        subj_pearsons.append(pearson_r(chrom_sig, s))

        # Save first window of first 4 subjects for waveform plot
        if len(waveform_samples) < 4 and i == 0:
            waveform_samples.append({
                "id":    subj_idx + 1,
                "chrom": chrom_sig.copy(),
                "gt":    s.copy(),
                "r":     pearson_r(chrom_sig, s),
                "c_bpm": c_bpm,
                "g_bpm": g_bpm,
            })

    if not subj_pred_bpms:
        continue

    pred_mean = np.mean(subj_pred_bpms)
    gt_mean   = np.mean(subj_gt_bpms)
    mae_subj  = abs(pred_mean - gt_mean)
    r_subj    = np.mean(subj_pearsons)

    subject_rows.append({
        "id":       subj_idx + 1,
        "gt_bpm":   gt_mean,
        "pred_bpm": pred_mean,
        "mae":      mae_subj,
        "pearson":  r_subj,
    })

    all_pred_bpm.extend(subj_pred_bpms)
    all_gt_bpm.extend(subj_gt_bpms)
    all_pearson.extend(subj_pearsons)


# ── Overall metrics ───────────────────────────────────────────────────
all_pred = np.array(all_pred_bpm)
all_gt   = np.array(all_gt_bpm)

mae_overall  = np.mean(np.abs(all_pred - all_gt))
rmse_overall = np.sqrt(np.mean((all_pred - all_gt)**2))
r_overall    = np.mean(all_pearson)

print("=" * 50)
print("     CHROM EVALUATION RESULTS — Val Set")
print("=" * 50)
print(f"  MAE      : {mae_overall:.2f} BPM")
print(f"  RMSE     : {rmse_overall:.2f} BPM")
print(f"  Pearson  : {r_overall:.4f}")
print(f"  Subjects : {len(subject_rows)}")
print("=" * 50)

print(f"\n{'ID':>4}  {'GT BPM':>8}  {'CHROM BPM':>10}  {'MAE':>6}  {'Pearson':>8}")
print("-" * 50)
for r in sorted(subject_rows, key=lambda x: x["mae"]):
    print(f"  {r['id']:>2}   {r['gt_bpm']:>7.2f}   {r['pred_bpm']:>9.2f}  "
          f"{r['mae']:>5.2f}   {r['pearson']:>7.4f}")

print(f"\nBest  MAE: {min(subject_rows, key=lambda x: x['mae'])['mae']:.2f} BPM")
print(f"Worst MAE: {max(subject_rows, key=lambda x: x['mae'])['mae']:.2f} BPM")

print(f"\n--- Comparison (once you run evaluate.py for DL) ---")
print(f"  CHROM MAE:           {mae_overall:.2f} BPM  ← this script")
print(f"  DeepPhysLSTM MAE:      5.62 BPM  ← from evaluate.py")
print(f"  Community CHROM ref:  ~7.0 BPM")


# ── Plot 1: Waveform — CHROM vs GT ───────────────────────────────────
n = len(waveform_samples)
fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
if n == 1:
    axes = [axes]

t = np.arange(128) / FPS

for ax, sample in zip(axes, waveform_samples):
    def norm(s):
        return (s - s.mean()) / (s.std() + 1e-8)

    ax.plot(t, norm(sample["gt"]),    color="#4CAF50", linewidth=1.5,
            label=f"Ground Truth PPG ({sample['g_bpm']:.1f} BPM)", alpha=0.9)
    ax.plot(t, norm(sample["chrom"]), color="#FF9800", linewidth=1.5,
            label=f"CHROM ({sample['c_bpm']:.1f} BPM)", linestyle="--", alpha=0.9)
    ax.set_title(f"Subject {sample['id']}  |  Pearson r = {sample['r']:.3f}", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])

plt.suptitle("CHROM vs Ground Truth PPG Signal",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
p1 = os.path.join(PLOT_PATH, "chrom_waveform.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {p1}")


# ── Plot 2: BPM scatter ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_gt, all_pred, alpha=0.5, color="#FF9800", s=30, label="Windows")

mn = min(min(all_gt), min(all_pred)) - 5
mx = max(max(all_gt), max(all_pred)) + 5
ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect prediction")
ax.fill_between([mn, mx], [mn-5, mx-5], [mn+5, mx+5],
                alpha=0.1, color="green", label="±5 BPM band")

ax.set_xlabel("Ground Truth BPM", fontsize=12)
ax.set_ylabel("CHROM Predicted BPM", fontsize=12)
ax.set_title(f"CHROM BPM Scatter — MAE={mae_overall:.2f}  RMSE={rmse_overall:.2f}  r={r_overall:.3f}",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([mn, mx])
ax.set_ylim([mn, mx])
ax.set_aspect("equal")

p2 = os.path.join(PLOT_PATH, "chrom_bpm_scatter.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {p2}")


# ── Plot 3: Per-subject MAE bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ids    = [r["id"]  for r in subject_rows]
maes   = [r["mae"] for r in subject_rows]
colors = ["#4CAF50" if m < 5 else "#FF9800" if m < 10 else "#F44336" for m in maes]

ax.bar(range(len(ids)), maes, color=colors, edgecolor="white")
ax.axhline(mae_overall, color="black", linestyle="--", linewidth=1.5,
           label=f"Mean MAE = {mae_overall:.2f} BPM")
ax.axhline(5, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
           label="5 BPM threshold")

ax.set_xticks(range(len(ids)))
ax.set_xticklabels([f"S{i}" for i in ids], fontsize=8)
ax.set_xlabel("Subject", fontsize=12)
ax.set_ylabel("MAE (BPM)", fontsize=12)
ax.set_title("CHROM Per-Subject MAE  |  Green<5  Orange<10  Red≥10", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

p3 = os.path.join(PLOT_PATH, "chrom_per_subject_mae.png")
plt.tight_layout()
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {p3}")

print(f"\nAll plots saved to: {PLOT_PATH}")
print("Done.")
