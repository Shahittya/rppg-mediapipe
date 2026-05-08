import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from dataset_fast import RPPGFastDataset
from models.deepphys.model import DeepPhysLSTM
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
#Pearson Loss
class NegPearsonLoss(nn.Module):
    def forward(self, pred, label):
        pred   = pred - pred.mean()
        label  = label - label.mean()
        num    = (pred * label).sum()
        denom  = torch.sqrt((pred**2).sum() * (label**2).sum()) + 1e-8
        return 1 - num / denom

torch.backends.cudnn.benchmark = True

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/content/drive/MyDrive/shared/FYP/processed_mediapipe"

dataset    = RPPGFastDataset(DATA_PATH)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

print(f"Device: {device} | Train: {train_size} | Val: {val_size}")

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


#Model 
model = DeepPhysLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

criterion     = NegPearsonLoss()
best_val_loss = float("inf")
EPOCHS        = 20

scaler = torch.amp.GradScaler('cuda')

# chunk size
CHUNK_SIZE = 3


# Training Loop
for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    #TRAIN 
    model.train()
    epoch_loss, count = 0.0, 0

    for batch_idx, (appearance, motion, signal) in enumerate(train_loader):
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx+1}/{len(train_loader)}")

        appearance = appearance.squeeze(0)
        motion     = motion.squeeze(0)
        signal     = signal.squeeze(0)

        optimizer.zero_grad()

        try:
            total_loss = 0
            num_chunks = 0

            with torch.amp.autocast('cuda'):
                for start in range(0, len(motion), CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, len(motion))

                    chunk_loss = 0

                    # KEEP TEMPORAL STRUCTURE (NO reshape)
                    for i in range(start, end):
                        a = appearance[i].float().to(device)   # (128, H, W, C)
                        m = motion[i].float().to(device)
                        s = signal[i].float().to(device)

                        pred = model(a, m)
                        chunk_loss += criterion(pred, s)

                    chunk_loss = chunk_loss / (end - start)
                    total_loss += chunk_loss
                    num_chunks += 1

                loss = total_loss / num_chunks

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            count += 1

        except Exception as e:
            print(f"  skip batch {batch_idx}: {e}")

        torch.cuda.empty_cache()

    train_avg = epoch_loss / max(count, 1)
    print(f"  Train loss: {train_avg:.4f}")


    #VALIDATION 
    model.eval()
    val_loss, val_count = 0.0, 0

    with torch.no_grad():
        for appearance, motion, signal in val_loader:

            appearance = appearance.squeeze(0)
            motion     = motion.squeeze(0)
            signal     = signal.squeeze(0)

            try:
                total_loss = 0
                num_chunks = 0

                with torch.amp.autocast('cuda'):
                    for start in range(0, len(motion), CHUNK_SIZE):
                        end = min(start + CHUNK_SIZE, len(motion))

                        chunk_loss = 0

                        for i in range(start, end):
                            a = appearance[i].float().to(device)
                            m = motion[i].float().to(device)
                            s = signal[i].float().to(device)

                            pred = model(a, m)
                            chunk_loss += criterion(pred, s)

                        chunk_loss = chunk_loss / (end - start)
                        total_loss += chunk_loss
                        num_chunks += 1

                val_loss += (total_loss / num_chunks).item()
                val_count += 1

            except Exception as e:
                print(f"  val skip: {e}")

    val_avg = val_loss / max(val_count, 1)
    print(f"  Val   loss: {val_avg:.4f}")

    scheduler.step(val_avg)

    if val_avg < best_val_loss:
        best_val_loss = val_avg
        torch.save(model.state_dict(), "best_model_mediapipe.pth")
        print(f"Best saved (val={val_avg:.4f})")

print(f"\nDone. Best val loss: {best_val_loss:.4f}")