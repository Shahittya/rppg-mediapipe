import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepPhysLSTM(nn.Module):
    def __init__(self):
        super(DeepPhysLSTM, self).__init__()

        # Motion stream — Tanh keeps negative motion values (darkening = pulse valley)
        self.motion_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )

        # Appearance stream — Sigmoid creates spatial attention mask [0, 1]
        self.appearance_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )

        # LSTM — input_size=256 because (2,2) pool gives 2*2*64=256 features
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # Final regression
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, appearance, motion):
        """
        appearance: (T, H, W, C)
        motion:     (T, H, W, C)
        """

        # Convert to (T, C, H, W)
        appearance = appearance.permute(0, 3, 1, 2)
        motion     = motion.permute(0, 3, 1, 2)

        # CNN feature extraction
        motion_feat     = self.motion_stream(motion)         # (T, 64, H, W)
        appearance_feat = self.appearance_stream(appearance) # (T, 64, H, W) attention mask

        # Attended fusion
        x = motion_feat * appearance_feat  # (T, 64, H, W)

        # (2,2) pool → keeps more spatial info → (T, 64, 2, 2)
        x = F.adaptive_avg_pool2d(x, (2, 2))
        x = x.reshape(x.size(0), -1)  # (T, 256)

        # Add batch dimension → (1, T, 256)
        x = x.unsqueeze(0)

        # LSTM → (1, T, 128)
        lstm_out, _ = self.lstm(x)

        # Remove batch → (T, 128)
        lstm_out = lstm_out.squeeze(0)

        # Final prediction → (T,)
        out = self.fc(lstm_out).squeeze(-1)

        return out
