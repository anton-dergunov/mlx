# -----------------------------------------------------------
# ✅ Import standard libraries
# -----------------------------------------------------------
import os
import glob
import random
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from transformers import WhisperProcessor, WhisperModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pickle

# -----------------------------------------------------------
# ✅ Config
# -----------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
DATA_ROOT = "/Users/anton/experiment_data/datasets/ravdess/"
CACHE_PATH = "emotion_full_hiddenstates_cache.pkl"
BATCH_SIZE = 16  # Smaller because sequence tensors are larger
EPOCHS = 30
LR = 1e-3

# -----------------------------------------------------------
# ✅ Prepare Whisper
# -----------------------------------------------------------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperModel.from_pretrained("openai/whisper-small")
model = model.eval().to(DEVICE)

# -----------------------------------------------------------
# ✅ Get all WAV files
# -----------------------------------------------------------
emotion_files = glob.glob(f"{DATA_ROOT}/**/*.wav", recursive=True)
print(f"Found {len(emotion_files)} files")

# -----------------------------------------------------------
# ✅ Emotion label map
# -----------------------------------------------------------
emotion_map = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

# -----------------------------------------------------------
# ✅ Try load cache
# -----------------------------------------------------------
if os.path.exists(CACHE_PATH):
    print(f"Loading cached embeddings from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    hidden_states_list = cached['hidden_states']
    labels = cached['labels']

else:
    print(f"No cache found. Extracting hidden states...")
    hidden_states_list = []
    labels = []

    for f in tqdm(emotion_files, desc="Extracting hidden states"):
        waveform, sr = torchaudio.load(f)
        # If stereo, convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze().numpy()

        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(torch.tensor(waveform)).squeeze().numpy()
            sr = 16000

        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            encoder_out = model.encoder(**inputs)
        states = encoder_out.last_hidden_state[0].cpu().numpy()  # shape: [seq_len, dim]

        code = int(os.path.basename(f).split("-")[2])
        label = emotion_map.get(code, "unknown")

        if label != "unknown":
            hidden_states_list.append(states)
            labels.append(label)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({'hidden_states': hidden_states_list, 'labels': labels}, f)
    print(f"Cached to {CACHE_PATH}")

print(f"Total clips cached: {len(hidden_states_list)}")

# -----------------------------------------------------------
# ✅ Encode labels
# -----------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(labels)
print(f"Label classes: {le.classes_}")

# -----------------------------------------------------------
# ✅ Example: build mean pooled version for now
# -----------------------------------------------------------
X_pooled = np.stack([np.mean(seq, axis=0) for seq in hidden_states_list])

# -----------------------------------------------------------
# ✅ Train/test split
# -----------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_pooled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# -----------------------------------------------------------
# ✅ Dataset
# -----------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = EmotionDataset(X_train, y_train)
val_ds = EmotionDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------------------------------
# ✅ Simple classifier
# -----------------------------------------------------------
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_pooled.shape[1]
num_classes = len(le.classes_)

model_cls = EmotionClassifier(input_dim, num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model_cls.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------
# ✅ Train loop
# -----------------------------------------------------------
for epoch in range(EPOCHS):
    model_cls.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model_cls(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # Validate
    model_cls.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(DEVICE)
            logits = model_cls(Xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f} - Val Macro F1: {f1:.4f}")
