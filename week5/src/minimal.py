import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm

from utils import get_device

# --------------------
# CONFIG
# --------------------
SAMPLE_RATE = 22050
NUM_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
DURATION = 4  # seconds
FIXED_LENGTH = SAMPLE_RATE * DURATION
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # for Mac

DEVICE = get_device()
print(f"Device: {DEVICE}")

# --------------------
# Dataset wrapper
# --------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, split, fold, is_train):
        self.ds = load_dataset("danavery/urbansound8K", split=split)
        self.ds = self.ds.filter(lambda x: x['fold'] != fold) if is_train else self.ds.filter(lambda x: x['fold'] == fold)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=NUM_MELS
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        waveform = torch.tensor(item['audio']['array'], dtype=torch.float32)
        sr = item['audio']['sampling_rate']

        if sr != SAMPLE_RATE:
            waveform = self.torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if waveform.size(0) < FIXED_LENGTH:
            pad_size = FIXED_LENGTH - waveform.size(0)
            waveform = F.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:FIXED_LENGTH]

        mel_spec = self.mel_transform(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
        )
        mel_spec = mel_spec.unsqueeze(0)

        label = item['classID']
        return mel_spec, label

# --------------------
# Simple CNN model
# --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
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

# --------------------
# Train & evaluate
# --------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm

# --------------------
# MAIN ENTRYPOINT
# --------------------
if __name__ == "__main__":
    fold_accuracies = []
    fold_f1s = []

    for fold in range(1, 11):
        print(f"=== Fold {fold} ===")

        train_ds = UrbanSoundDataset("train", fold, is_train=True)
        test_ds = UrbanSoundDataset("train", fold, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = SimpleCNN().to(DEVICE)
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

        acc, f1, cm = evaluate(model, test_loader)
        print(f"Fold {fold} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        fold_accuracies.append(acc)
        fold_f1s.append(f1)

    print("=== 10-fold CV Results ===")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Mean Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
