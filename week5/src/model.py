import torch.nn as nn
import torch.nn.functional as F


# FIXME Load these params from config and avoid duplication
SAMPLE_RATE = 22050
NUM_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
DURATION = 4  # seconds
FIXED_LENGTH = SAMPLE_RATE * DURATION


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # TODO Also try nn.Conv1d
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * (NUM_MELS // 8) * (int(FIXED_LENGTH / HOP_LENGTH) // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
