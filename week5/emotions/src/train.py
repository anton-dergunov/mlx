# -----------------------------------------------------------
# ✅ Libraries
# -----------------------------------------------------------
import os
import glob
import random
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from transformers import WhisperProcessor, WhisperModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pickle

from model import MLPClassifier, TinyTransformerClassifier


# -----------------------------------------------------------
# ✅ Config
# -----------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
DATA_ROOT = "/Users/anton/experiment_data/datasets/ravdess/"
CACHE_PATH = "emotion_full_hiddenstates_cache_with_actors.pkl"
BATCH_SIZE = 8  # lower batch for transformer head
EPOCHS = 20
LR = 1e-3

# Pick: "mlp" or "transformer"
MODEL_TYPE = "mlp"

# -----------------------------------------------------------
# ✅ Whisper model
# -----------------------------------------------------------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperModel.from_pretrained("openai/whisper-small")
model = model.eval().to(DEVICE)

# -----------------------------------------------------------
# ✅ All files
# -----------------------------------------------------------
emotion_files = glob.glob(f"{DATA_ROOT}/**/*.wav", recursive=True)
print(f"Found {len(emotion_files)} files")

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
# ✅ Cache: hidden states, labels, actor IDs
# -----------------------------------------------------------
if os.path.exists(CACHE_PATH):
    print(f"Loading cached data from {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    hidden_states_list = cached['hidden_states']
    labels = cached['labels']
    actor_ids = cached['actors']
else:
    print(f"No cache found. Extracting...")
    hidden_states_list = []
    labels = []
    actor_ids = []

    for f in tqdm(emotion_files, desc="Extracting hidden states"):
        waveform, sr = torchaudio.load(f)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze().numpy()

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(torch.tensor(waveform)).squeeze().numpy()
            sr = 16000

        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            encoder_out = model.encoder(**inputs)
        states = encoder_out.last_hidden_state[0].cpu().numpy()  # [seq_len, hidden_dim]

        code = int(os.path.basename(f).split("-")[2])
        label = emotion_map.get(code, "unknown")

        actor_id = int(os.path.basename(f).split(".")[0].split("-")[6])

        if label != "unknown":
            hidden_states_list.append(states)
            labels.append(label)
            actor_ids.append(actor_id)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            'hidden_states': hidden_states_list,
            'labels': labels,
            'actors': actor_ids
        }, f)
    print(f"Cached to {CACHE_PATH}")

print(f"Total samples: {len(hidden_states_list)}")

# -----------------------------------------------------------
# ✅ Encode labels
# -----------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(labels)
actors = np.array(actor_ids)

print(f"Label classes: {le.classes_}")

# -----------------------------------------------------------
# ✅ For MLP baseline: use mean pooling
# -----------------------------------------------------------
X_pooled = np.stack([np.mean(seq, axis=0) for seq in hidden_states_list])

# -----------------------------------------------------------
# ✅ Split by actor
# -----------------------------------------------------------
val_mask = np.isin(actors, [1, 2])  # Actor_01 and Actor_02 for val

if MODEL_TYPE == "mlp":
    X_train = X_pooled[~val_mask]
    y_train = y[~val_mask]

    X_val = X_pooled[val_mask]
    y_val = y[val_mask]
else:
    # For transformer head: keep full sequence
    X_train = [hidden_states_list[i] for i in range(len(hidden_states_list)) if not val_mask[i]]
    y_train = y[~val_mask]

    X_val = [hidden_states_list[i] for i in range(len(hidden_states_list)) if val_mask[i]]
    y_val = y[val_mask]

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# -----------------------------------------------------------
# ✅ Datasets
# -----------------------------------------------------------
class MLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerDataset(Dataset):
    def __init__(self, seqs, y):
        self.seqs = seqs
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.float32)  # [seq_len, dim]
        label = self.y[idx]
        return seq, label

# Collate function for padding
def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in seqs]
    max_len = max(lengths)
    padded = torch.zeros(len(seqs), max_len, seqs[0].shape[1])
    for i, seq in enumerate(seqs):
        padded[i, :seq.shape[0], :] = seq
    return padded, torch.tensor(labels), torch.tensor(lengths)

if MODEL_TYPE == "mlp":
    train_ds = MLPDataset(X_train, y_train)
    val_ds = MLPDataset(X_val, y_val)
    collate = None
else:
    train_ds = TransformerDataset(X_train, y_train)
    val_ds = TransformerDataset(X_val, y_val)
    collate = collate_fn

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

input_dim = X_pooled.shape[1]
num_classes = len(le.classes_)

if MODEL_TYPE == "mlp":
    model_cls = MLPClassifier(input_dim, num_classes).to(DEVICE)
else:
    model_cls = TinyTransformerClassifier(input_dim, num_classes).to(DEVICE)

optimizer = torch.optim.Adam(model_cls.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------
# ✅ Train loop
# -----------------------------------------------------------
for epoch in range(EPOCHS):
    model_cls.train()
    total_loss = 0

    for batch in train_loader:
        if MODEL_TYPE == "mlp":
            Xb, yb = batch
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model_cls(Xb)
        else:
            Xb, yb, lengths = batch
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model_cls(Xb, lengths)

        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # Validate
    model_cls.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            if MODEL_TYPE == "mlp":
                Xb, yb = batch
                Xb = Xb.to(DEVICE)
                logits = model_cls(Xb)
            else:
                Xb, yb, lengths = batch
                Xb = Xb.to(DEVICE)
                logits = model_cls(Xb, lengths)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

model_path = f"{MODEL_TYPE}.pt"
torch.save(model_cls.state_dict(), model_path)
print(f"Model saved to {model_path}")
