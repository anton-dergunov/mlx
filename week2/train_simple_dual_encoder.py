import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from gensim.models import KeyedVectors
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import re

from util import get_device


def get_dataset():
    ds = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        cache_dir="~/experiment_data/datasets/ms_marco",
        split="train")

    rows = []
    for entry in ds:
        q = entry["query"]
        passages = entry["passages"]
        for is_selected, text in zip(passages['is_selected'], passages['passage_text']):
            if is_selected == 1:
                rows.append((q, text))

    all_passages = []
    for p in ds["passages"]:
        all_passages.extend(p["passage_text"])

    class TripletDataset(Dataset):
        def __init__(self, rows, all_passages):
            self.rows = rows
            self.all_passages = all_passages
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            q, pos = self.rows[idx]
            neg = pos
            while neg == pos:
                neg = random.choice(self.all_passages)
            return {"query": q, "pos": pos, "neg": neg}
    
    return TripletDataset(rows, all_passages)

# Load pretrained:
# Download the data from https://code.google.com/archive/p/word2vec/
# TODO Replace with configurable path

print("Loading pretrained word2vec model...")
w2v = KeyedVectors.load_word2vec_format("/Users/anton/experiment_data/datasets/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", binary=True)

class AvgW2VEncoder(nn.Module):
    def __init__(self, w2v_model, hidden_dim=128, device="cpu"):
        super().__init__()
        self.w2v = w2v_model
        self.embedding_dim = w2v_model.vector_size
        self.proj = nn.Linear(self.embedding_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.device = device

    def forward(self, texts):
        embs = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            vecs = [self.w2v[w] for w in words if w in self.w2v]
            if len(vecs)==0:
                vec = torch.zeros(self.embedding_dim, device=self.device)
            else:
                vec = torch.from_numpy(np.array(vecs)).to(self.device).mean(dim=0)
            embs.append(vec)
        embs = torch.stack(embs)  # (B, E)
        out = self.norm(self.proj(embs))  # (B, H)
        return out

device = get_device()
print(f"Using device: {device}")

encoder = AvgW2VEncoder(w2v_model=w2v, hidden_dim=128, device=device).to(device)
optimizer = optim.Adam(encoder.proj.parameters(), lr=1e-4)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

def collate_fn(batch):
    return {
        "query": [item["query"] for item in batch],
        "pos": [item["pos"] for item in batch],
        "neg": [item["neg"] for item in batch],
    }

print("Loading dataset...")
loader = DataLoader(get_dataset(), batch_size=32, shuffle=True, collate_fn=collate_fn)

print("Starting training...")
for epoch in range(3):
    epoch_loss = 0.0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        q, pos, neg = batch["query"], batch["pos"], batch["neg"]
        q_emb = encoder(q)
        pos_emb = encoder(pos)
        neg_emb = encoder(neg)

        loss = triplet_loss(q_emb, pos_emb, neg_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: avg loss = {epoch_loss/len(loader):.4f}")
