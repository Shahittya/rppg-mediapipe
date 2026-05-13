"""
evaluate.py  — Fixed version

Place in: rppg-mediapipe-main/  (same folder as train.py)

Fixes over previous version:
  1. Uses calculate_heart_rate() with prev_bpm continuity tracking
  2. Correct BPM range 0.7-2.5 Hz (42-150 BPM)
  3. Full waveform + scatter + per-subject MAE plots
  4. Plots saved to Google Drive

Run: !python evaluate.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed in Colab
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset_fast import RPPGFastDataset
from models.deepphys.model import DeepPhysLSTM
from utils.heart_rate import calculate_heart_rate

DATA_PATH  = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"
MODEL_PATH = "best_model_mediapipe.pth"
PLOT_PATH  = "/content/drive/MyDrive/shared/FYP/eval_plots"
FPS        = 30
SEED       = 42

os.makedirs(PLOT_PATH, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")



def pearson_r(pred, gt):
    p = pred - pred.mean()
    g = gt   - gt.mean()
    denom = np.sqrt((p**2).sum() * (g**2).sum()) + 1e-8
    return float((p * g).sum() / denom)


#Load dataset + val split
dataset    = RPPGFastDataset(DATA_PATH)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

_, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
print(f"Val subjects: {len(val_ds)}\n")


#Load model 
model = DeepPhysLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded: {MODEL_PATH}\n")


# Evaluate
all_pred_bpm = []
all_gt_bpm   = []
all_pearson  = []
subject_rows = []
waveform_samples = []   # for plots — store first window of first 4 subjects

with torch.no_grad():
    for subj_idx, (appearance, motion, signal) in enumerate(val_loader):
        appearance = appearance.squeeze(0)
        motion     = motion.squeeze(0)
        signal     = signal.squeeze(0).numpy()

        subj_pred_bpms  = []
        subj_gt_bpms    = []
        subj_pearsons   = []
        prev_bpm        = None   # ← continuity tracking per subject

        for i in range(len(motion)):
            a = appearance[i].float().to(device)
            m = motion[i].float().to(device)
            s = signal[i]

            with torch.amp.autocast('cuda'):
                pred = model(a, m).cpu().numpy()

            # ── BPM with continuity tracking 
            p_bpm, _, _ = calculate_heart_rate(pred, FPS, prev_bpm)
            g_bpm, _, _ = calculate_heart_rate(s,    FPS, None)

            if p_bpm <= 0 or g_bpm <= 0:
                continue

            prev_bpm = p_bpm   # update for next window

            subj_pred_bpms.append(p_bpm)
            subj_gt_bpms.append(g_bpm)
            subj_pearsons.append(pearson_r(pred, s))

            # Save first window for waveform plot
            if len(waveform_samples) < 4 and i == 0:
                waveform_samples.append({
                    "id":   subj_idx + 1,
                    "pred": pred.copy(),
                    "gt":   s.copy(),
                    "r":    pearson_r(pred, s),
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
rmse_overall = np.sqrt(np.mean((all_pred - all_gt) ** 2))
r_overall    = np.mean(all_pearson)

print("=" * 50)
print("        EVALUATION RESULTS — Val Set")
print("=" * 50)
print(f"  MAE      : {mae_overall:.2f} BPM   ← report this")
print(f"  RMSE     : {rmse_overall:.2f} BPM")
print(f"  Pearson  : {r_overall:.4f}")
print(f"  Subjects : {len(subject_rows)}")
print("=" * 50)

print(f"\n{'ID':>4}  {'GT BPM':>8}  {'Pred BPM':>9}  {'MAE':>6}  {'Pearson':>8}")
print("-" * 47)
for r in sorted(subject_rows, key=lambda x: x["mae"]):
    print(f"  {r['id']:>2}   {r['gt_bpm']:>7.2f}   {r['pred_bpm']:>8.2f}  "
          f"{r['mae']:>5.2f}   {r['pearson']:>7.4f}")

print(f"\nBest  MAE: {min(subject_rows, key=lambda x: x['mae'])['mae']:.2f} BPM")
print(f"Worst MAE: {max(subject_rows, key=lambda x: x['mae'])['mae']:.2f} BPM")

print(f"\n--- Community benchmark comparison ---")
print(f"  Your MAE:              {mae_overall:.2f} BPM")
print(f"  DeepPhys (UBFC):      ~11.0 BPM")
print(f"  EfficientPhys (UBFC):  ~5.3 BPM")
print(f"  CHROM (traditional):   ~7.0 BPM")


# Plot 1: Waveform comparison 
n = len(waveform_samples)
fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
if n == 1:
    axes = [axes]

t = np.arange(128) / FPS

for ax, sample in zip(axes, waveform_samples):
    pred_n = (sample["pred"] - sample["pred"].mean()) / (sample["pred"].std() + 1e-8)
    gt_n   = (sample["gt"]   - sample["gt"].mean())   / (sample["gt"].std()   + 1e-8)

    ax.plot(t, gt_n,   color="#2196F3", linewidth=1.5,
            label="Ground Truth PPG", alpha=0.9)
    ax.plot(t, pred_n, color="#F44336", linewidth=1.5,
            label="Predicted Signal", linestyle="--", alpha=0.9)
    ax.set_title(f"Subject {sample['id']} | Pearson r = {sample['r']:.3f}", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])

plt.suptitle("Predicted vs Ground Truth rPPG Signal",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
p1 = os.path.join(PLOT_PATH, "waveform_comparison.png")
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {p1}")


# Plot 2: BPM scatter 
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(all_gt, all_pred, alpha=0.5, color="#673AB7", s=30, label="Windows")

mn = min(min(all_gt), min(all_pred)) - 5
mx = max(max(all_gt), max(all_pred)) + 5
ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect prediction")
ax.fill_between([mn, mx], [mn-5, mx-5], [mn+5, mx+5],
                alpha=0.1, color="green", label="±5 BPM band")

ax.set_xlabel("Ground Truth BPM", fontsize=12)
ax.set_ylabel("Predicted BPM", fontsize=12)
ax.set_title(f"BPM Scatter — MAE={mae_overall:.2f}  RMSE={rmse_overall:.2f}  r={r_overall:.3f}",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([mn, mx])
ax.set_ylim([mn, mx])
ax.set_aspect("equal")

p2 = os.path.join(PLOT_PATH, "bpm_scatter.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {p2}")


# Plot 3: Per-subject MAE bar chart
fig, ax = plt.subplots(figsize=(12, 5))

ids    = [r["id"]  for r in subject_rows]
maes   = [r["mae"] for r in subject_rows]
colors = ["#4CAF50" if m < 5 else "#FF9800" if m < 10 else "#F44336" for m in maes]

ax.bar(range(len(ids)), maes, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(mae_overall, color="black", linestyle="--", linewidth=1.5,
           label=f"Mean MAE = {mae_overall:.2f} BPM")
ax.axhline(5, color="green", linestyle=":", linewidth=1.2, alpha=0.7,
           label="5 BPM threshold")

ax.set_xticks(range(len(ids)))
ax.set_xticklabels([f"S{i}" for i in ids], fontsize=8)
ax.set_xlabel("Subject", fontsize=12)
ax.set_ylabel("MAE (BPM)", fontsize=12)
ax.set_title("Per-Subject MAE  |  Green < 5 BPM  |  Orange < 10 BPM  |  Red ≥ 10 BPM",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

p3 = os.path.join(PLOT_PATH, "per_subject_mae.png")
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {p3}")

print(f"\nAll 3 plots saved to Google Drive: {PLOT_PATH}")
print("Done.")
