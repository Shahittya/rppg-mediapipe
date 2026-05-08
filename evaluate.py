

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset_fast import RPPGFastDataset
from models.deepphys.model import DeepPhysLSTM

# ── same seed as training so val split is identical 
torch.manual_seed(42)
np.random.seed(42)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"
MODEL_PATH = "best_model_mediapipe.pth"
FPS        = 30


#  helpers 

def get_bpm(signal, fps=FPS, n_fft=512):
    """FFT-based BPM estimation with zero-padding for resolution."""
    freqs = np.fft.rfftfreq(n_fft, d=1/fps)
    power = np.abs(np.fft.rfft(signal, n=n_fft)) ** 2
    mask  = (freqs >= 0.7) & (freqs <= 3.0)
    if not mask.any():
        return None
    return freqs[mask][power[mask].argmax()] * 60


def pearson(pred, gt):
    pred = pred - pred.mean()
    gt   = gt   - gt.mean()
    denom = np.sqrt((pred**2).sum() * (gt**2).sum()) + 1e-8
    return (pred * gt).sum() / denom


# load dataset + val spli

dataset    = RPPGFastDataset(DATA_PATH)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
print(f"Val subjects: {len(val_ds)}")


# load model 

model = DeepPhysLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}\n")


# evaluate 

all_pred_bpm = []
all_gt_bpm   = []
all_pearson  = []

subject_results = []

with torch.no_grad():
    for subj_idx, (appearance, motion, signal) in enumerate(val_loader):
        appearance = appearance.squeeze(0)   # (N, T, H, W, 3)
        motion     = motion.squeeze(0)
        signal     = signal.squeeze(0)       # (N, T)

        subj_pred_bpm = []
        subj_gt_bpm   = []
        subj_pearson  = []

        for i in range(len(motion)):
            a = appearance[i].float().to(device)
            m = motion[i].float().to(device)
            s = signal[i].numpy()              # ground truth window

            with torch.amp.autocast('cuda'):
                pred = model(a, m).cpu().numpy()   # predicted signal

            pred_bpm = get_bpm(pred)
            gt_bpm   = get_bpm(s)

            if pred_bpm is None or gt_bpm is None:
                continue

            r = pearson(pred, s)

            subj_pred_bpm.append(pred_bpm)
            subj_gt_bpm.append(gt_bpm)
            subj_pearson.append(r)

        if not subj_pred_bpm:
            continue

        subj_pred_mean = np.mean(subj_pred_bpm)
        subj_gt_mean   = np.mean(subj_gt_bpm)
        subj_mae       = abs(subj_pred_mean - subj_gt_mean)
        subj_r         = np.mean(subj_pearson)

        subject_results.append({
            "subject": subj_idx,
            "pred_bpm": subj_pred_mean,
            "gt_bpm":   subj_gt_mean,
            "mae":      subj_mae,
            "pearson":  subj_r,
        })

        all_pred_bpm.extend(subj_pred_bpm)
        all_gt_bpm.extend(subj_gt_bpm)
        all_pearson.extend(subj_pearson)


#results
all_pred = np.array(all_pred_bpm)
all_gt   = np.array(all_gt_bpm)

mae_val  = np.mean(np.abs(all_pred - all_gt))
rmse_val = np.sqrt(np.mean((all_pred - all_gt) ** 2))
r_val    = np.mean(all_pearson)

print("=" * 45)
print("       EVALUATION RESULTS (Val Set)")
print("=" * 45)
print(f"  MAE:     {mae_val:.2f} BPM   ← main metric")
print(f"  RMSE:    {rmse_val:.2f} BPM")
print(f"  Pearson: {r_val:.4f}")
print(f"  Subjects evaluated: {len(subject_results)}")
print("=" * 45)

print("\n--- Per-subject breakdown ---")
print(f"{'Subj':>5}  {'GT BPM':>8}  {'Pred BPM':>9}  {'MAE':>6}  {'Pearson':>8}")
print("-" * 45)
for r in sorted(subject_results, key=lambda x: x["mae"]):
    print(f"  {r['subject']:>3}   {r['gt_bpm']:>7.2f}   {r['pred_bpm']:>8.2f}  "
          f"{r['mae']:>5.2f}   {r['pearson']:>7.4f}")

print(f"\nBest  MAE subject: {min(subject_results, key=lambda x: x['mae'])['mae']:.2f} BPM")
print(f"Worst MAE subject: {max(subject_results, key=lambda x: x['mae'])['mae']:.2f} BPM")

print("\n--- Community benchmark comparison ---")
print(f"  Your MAE:              {mae_val:.2f} BPM")
print(f"  DeepPhys (UBFC):      ~11.0 BPM")
print(f"  EfficientPhys (UBFC): ~5.3  BPM")
print(f"  CHROM (traditional):  ~7.0  BPM")
