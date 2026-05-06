import os
import numpy as np
from torch.utils.data import Dataset


class RPPGFastDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            appearance = data["appearance"]   # (N, T, H, W, 3)
            motion     = data["motion"]       # (N, T, H, W, 3)
            signal     = data["signal"]       # (N, T)

        signal     = signal.astype(np.float32)
        appearance = appearance.astype(np.float32)
        motion     = motion.astype(np.float32)

        # Appearance: [0,1] → [-1,1]  (centers distribution for conv)
        appearance = (appearance - 0.5) / 0.5

        # Motion: amplify weak pulse signal
        # Use *10 not *100 — BatchNorm inside model handles scale from here
        motion = motion * 10.0

        # Safety: remove any NaN/Inf
        if not np.isfinite(signal).all():
            signal = np.nan_to_num(signal)
        if not np.isfinite(appearance).all():
            appearance = np.nan_to_num(appearance)
        if not np.isfinite(motion).all():
            motion = np.nan_to_num(motion)

        return appearance, motion, signal
