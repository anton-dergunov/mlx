import torch.nn as nn
import torch.nn.functional as F


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
