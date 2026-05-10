"""
evaluate_comparison.py

Runs CHROM and your DL model on the SAME val subjects.
Produces:
  - Side-by-side comparison table (Subject | GT | CHROM | DL)
  - Overall metrics table (MAE, RMSE, Pearson)
  - 3 comparison plots saved to Google Drive

Place in: rppg-mediapipe-main/  (same folder as train.py)
Run:      !python evaluate_comparison.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset_fast import RPPGFastDataset
from models.deepphys.model import DeepPhysLSTM
from utils.chrom import chrom_method
from utils.heart_rate import calculate_heart_rate

# ── Config ────────────────────────────────────────────────────────────
DATA_PATH  = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"
MODEL_PATH = "best_model_mediapipe.pth"
PLOT_PATH  = "/content/drive/MyDrive/shared/FYP/eval_plots"
FPS        = 30
SEED       = 42

os.makedirs(PLOT_PATH, exist_ok=True)

# ── Reproducible split — identical to training ────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Helper ────────────────────────────────────────────────────────────

def pearson_r(pred, gt):
    p = pred - pred.mean()
    g = gt   - gt.mean()
    denom = np.sqrt((p**2).sum() * (g**2).sum()) + 1e-8
    return float((p * g).sum() / denom)


def get_bpm_from_signal(signal, prev_bpm=None):
    bpm, _, _ = calculate_heart_rate(signal, FPS, prev_bpm)
    return bpm if bpm > 0 else None


def chrom_from_appearance(appearance_window):
    """
    appearance_window: (T, H, W, 3) float32 in [-1, 1]
    Convert back to [0,1], spatial average per frame → RGB timeseries → CHROM
    """
    # Reverse dataset_fast normalization: [-1,1] → [0,1]
    app = (appearance_window * 0.5) + 0.5          # (T, H, W, 3)
    app = np.clip(app, 0, 1)

    # Spatial average per frame → (T, 3)
    r = app[:, :, :, 0].mean(axis=(1, 2))
    g = app[:, :, :, 1].mean(axis=(1, 2))
    b = app[:, :, :, 2].mean(axis=(1, 2))

    return chrom_method(r, g, b, fs=FPS)           # (T,)


# ── Load dataset + val split ──────────────────────────────────────────
dataset    = RPPGFastDataset(DATA_PATH)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

_, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
print(f"Val subjects: {len(val_ds)}\n")


# ── Load DL model ─────────────────────────────────────────────────────
model = DeepPhysLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded: {MODEL_PATH}\n")


# ── Run both methods on val set ───────────────────────────────────────
rows = []           # per-subject results
chrom_all_pred = []
chrom_all_gt   = []
chrom_all_r    = []
dl_all_pred    = []
dl_all_gt      = []
dl_all_r       = []

# for waveform plot — store one subject
plot_sample = None

with torch.no_grad():
    for subj_idx, (appearance, motion, signal) in enumerate(val_loader):
        appearance_np = appearance.squeeze(0).numpy()  # (N, T, H, W, 3)
        motion_t      = motion.squeeze(0)
        signal_np     = signal.squeeze(0).numpy()      # (N, T)

        chrom_bpms  = []
        dl_bpms     = []
        gt_bpms     = []
        chrom_rs    = []
        dl_rs       = []

        dl_prev_bpm    = None
        chrom_prev_bpm = None

        for i in range(len(motion_t)):
            s = signal_np[i]                           # GT signal window
            a_np = appearance_np[i]                    # (T, H, W, 3) in [-1,1]

            # ── CHROM ─────────────────────────────────────────────────
            chrom_sig = chrom_from_appearance(a_np)
            c_bpm = get_bpm_from_signal(chrom_sig, chrom_prev_bpm)
            if c_bpm:
                chrom_prev_bpm = c_bpm

            # ── DL model ──────────────────────────────────────────────
            a_t = motion_t[i].float().to(device)   # motion for model
            m_t = motion_t[i].float().to(device)
            # Note: model takes appearance + motion
            a_in = torch.tensor(a_np, dtype=torch.float32).to(device)
            with torch.amp.autocast('cuda'):
                dl_sig = model(a_in, m_t).cpu().numpy()
            d_bpm = get_bpm_from_signal(dl_sig, dl_prev_bpm)
            if d_bpm:
                dl_prev_bpm = d_bpm

            # ── GT BPM ────────────────────────────────────────────────
            g_bpm = get_bpm_from_signal(s)

            if not (c_bpm and d_bpm and g_bpm):
                continue

            chrom_bpms.append(c_bpm)
            dl_bpms.append(d_bpm)
            gt_bpms.append(g_bpm)
            chrom_rs.append(pearson_r(chrom_sig, s))
            dl_rs.append(pearson_r(dl_sig, s))

            # Save one sample for waveform plot
            if plot_sample is None and i == 0:
                plot_sample = {
                    "id":        subj_idx + 1,
                    "gt":        s.copy(),
                    "chrom":     chrom_sig.copy(),
                    "dl":        dl_sig.copy(),
                    "chrom_r":   pearson_r(chrom_sig, s),
                    "dl_r":      pearson_r(dl_sig, s),
                }

        if not gt_bpms:
            continue

        gt_mean    = np.mean(gt_bpms)
        chrom_mean = np.mean(chrom_bpms)
        dl_mean    = np.mean(dl_bpms)

        rows.append({
            "id":        subj_idx + 1,
            "gt_bpm":    gt_mean,
            "chrom_bpm": chrom_mean,
            "dl_bpm":    dl_mean,
            "chrom_mae": abs(chrom_mean - gt_mean),
            "dl_mae":    abs(dl_mean    - gt_mean),
            "chrom_r":   np.mean(chrom_rs),
            "dl_r":      np.mean(dl_rs),
        })

        chrom_all_pred.extend(chrom_bpms)
        dl_all_pred.extend(dl_bpms)
        chrom_all_gt.extend(gt_bpms)
        dl_all_gt.extend(gt_bpms)
        chrom_all_r.extend(chrom_rs)
        dl_all_r.extend(dl_rs)


# ── Overall metrics ───────────────────────────────────────────────────
def metrics(pred, gt, rs):
    pred, gt = np.array(pred), np.array(gt)
    return {
        "mae":     np.mean(np.abs(pred - gt)),
        "rmse":    np.sqrt(np.mean((pred - gt)**2)),
        "pearson": np.mean(rs),
    }

chrom_m = metrics(chrom_all_pred, chrom_all_gt, chrom_all_r)
dl_m    = metrics(dl_all_pred,    dl_all_gt,    dl_all_r)


# ── Print results ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("        CHROM vs Deep Learning — Val Set Comparison")
print("=" * 65)
print(f"\n{'Method':<20} {'MAE (BPM)':>10} {'RMSE (BPM)':>11} {'Pearson':>9}")
print("-" * 55)
print(f"{'CHROM':<20} {chrom_m['mae']:>10.2f} {chrom_m['rmse']:>11.2f} {chrom_m['pearson']:>9.4f}")
print(f"{'DeepPhysLSTM (Ours)':<20} {dl_m['mae']:>10.2f} {dl_m['rmse']:>11.2f} {dl_m['pearson']:>9.4f}")
print("-" * 55)
print(f"\n  MAE improvement over CHROM: {chrom_m['mae'] - dl_m['mae']:+.2f} BPM")

print(f"\n{'Sub':>4}  {'GT BPM':>8}  {'CHROM':>8}  {'CHROM MAE':>10}  {'DL BPM':>8}  {'DL MAE':>8}")
print("-" * 60)
for r in sorted(rows, key=lambda x: x["id"]):
    winner = "✓" if r["dl_mae"] < r["chrom_mae"] else " "
    print(f"  {r['id']:>2}   {r['gt_bpm']:>7.2f}   {r['chrom_bpm']:>7.2f}   "
          f"{r['chrom_mae']:>9.2f}   {r['dl_bpm']:>7.2f}   {r['dl_mae']:>7.2f}  {winner}")

dl_wins    = sum(1 for r in rows if r["dl_mae"] < r["chrom_mae"])
chrom_wins = sum(1 for r in rows if r["chrom_mae"] <= r["dl_mae"])
print(f"\nDL better on {dl_wins}/{len(rows)} subjects")
print(f"CHROM better on {chrom_wins}/{len(rows)} subjects")


# ── Plot 1: Method comparison bar chart ──────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x   = np.arange(len(rows))
w   = 0.35
ids = [r["id"] for r in rows]

b1 = ax.bar(x - w/2, [r["chrom_mae"] for r in rows],
            w, label="CHROM", color="#FF9800", alpha=0.85)
b2 = ax.bar(x + w/2, [r["dl_mae"] for r in rows],
            w, label="DeepPhysLSTM (Ours)", color="#2196F3", alpha=0.85)

ax.axhline(chrom_m["mae"], color="#FF9800", linestyle="--", linewidth=1.5,
           label=f"CHROM mean MAE = {chrom_m['mae']:.2f}")
ax.axhline(dl_m["mae"],    color="#2196F3", linestyle="--", linewidth=1.5,
           label=f"DL mean MAE = {dl_m['mae']:.2f}")

ax.set_xticks(x)
ax.set_xticklabels([f"S{i}" for i in ids], fontsize=8)
ax.set_xlabel("Subject", fontsize=12)
ax.set_ylabel("MAE (BPM)", fontsize=12)
ax.set_title("Per-Subject MAE: CHROM vs DeepPhysLSTM", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

p1 = os.path.join(PLOT_PATH, "chrom_vs_dl_bar.png")
plt.tight_layout()
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {p1}")


# ── Plot 2: Overall metrics comparison ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metrics_names = ["MAE (BPM)", "RMSE (BPM)", "Pearson"]
chrom_vals    = [chrom_m["mae"], chrom_m["rmse"], chrom_m["pearson"]]
dl_vals       = [dl_m["mae"],    dl_m["rmse"],    dl_m["pearson"]]
colors        = [["#FF9800", "#2196F3"]] * 3

for ax, name, cv, dv in zip(axes, metrics_names, chrom_vals, dl_vals):
    bars = ax.bar(["CHROM", "DeepPhysLSTM\n(Ours)"], [cv, dv],
                  color=["#FF9800", "#2196F3"], alpha=0.85, width=0.5)
    for bar, val in zip(bars, [cv, dv]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(name, fontsize=12)
    ax.set_ylim(0, max(cv, dv) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Overall Metrics: CHROM vs DeepPhysLSTM", fontsize=13, fontweight="bold")
plt.tight_layout()
p2 = os.path.join(PLOT_PATH, "chrom_vs_dl_metrics.png")
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {p2}")


# ── Plot 3: Waveform — GT vs CHROM vs DL ─────────────────────────────
if plot_sample:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    t = np.arange(128) / FPS

    def norm(s):
        return (s - s.mean()) / (s.std() + 1e-8)

    axes[0].plot(t, norm(plot_sample["gt"]),    color="#4CAF50", linewidth=1.5)
    axes[0].set_title(f"Subject {plot_sample['id']} — Ground Truth PPG", fontsize=11)

    axes[1].plot(t, norm(plot_sample["chrom"]), color="#FF9800", linewidth=1.5)
    axes[1].set_title(f"CHROM Signal  |  Pearson r = {plot_sample['chrom_r']:.3f}", fontsize=11)

    axes[2].plot(t, norm(plot_sample["dl"]),    color="#2196F3", linewidth=1.5)
    axes[2].set_title(f"DeepPhysLSTM  |  Pearson r = {plot_sample['dl_r']:.3f}", fontsize=11)

    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalised amplitude")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, t[-1]])

    plt.suptitle("Signal Comparison: GT vs CHROM vs DeepPhysLSTM",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = os.path.join(PLOT_PATH, "waveform_three_way.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {p3}")

print(f"\nAll plots saved to: {PLOT_PATH}")
print("Done.")
