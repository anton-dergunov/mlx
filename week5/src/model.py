import torch
import torch.nn as nn
import torch.nn.functional as F
import math


DEFAULT_SAMPLE_RATE = 22050
DEFAULT_NUM_MELS = 64
DEFAULT_HOP_LENGTH = 512
DEFAULT_MAX_DURATION = 4  # seconds
DEFAULT_FIXED_LENGTH = DEFAULT_SAMPLE_RATE * DEFAULT_MAX_DURATION


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_mels=DEFAULT_NUM_MELS,
        fixed_length=DEFAULT_FIXED_LENGTH,
        hop_length=DEFAULT_HOP_LENGTH,
        num_classes=10,
    ):
        super().__init__()
        # TODO Also try nn.Conv1d
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * (num_mels // 8) * (int(fixed_length / hop_length) // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # TODO What if a clip is a bit longer than 4 seconds?
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # TODO Clarify div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        # TODO What does register_buffer do? Get clarity
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class AudioTransformer(nn.Module):
    def __init__(self,
                 n_mels=64,  # FIXME Provide and use this argument
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 num_classes=10):
        super().__init__()

        # 2D conv frontend: [B, 1, freq, time] -> [B, C, freq, time]
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),  # both time and freq halved
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.proj = nn.Linear(128 * 8, d_model)

        # Positional encoding (sinusoidal, not learned)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True  # easier for us
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: [B, 1, n_mels, time]
        """
        # Conv frontend
        x = self.conv(x)  # [B, C, F, T]

        B, C, F, T = x.shape

        # Flatten freq dim to channels: treat each time step as a token
        x = x.permute(0, 3, 1, 2)  # [B, T, C, F]
        x = x.flatten(2)  # [B, T, C*F] = [B, T, d_model]

        x = self.proj(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer_encoder(x)  # [B, T, d_model]

        # Pool over time dim (mean)
        x = x.mean(dim=1)  # [B, d_model]

        return self.classifier(x)
