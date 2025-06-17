import os
import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ------------- Configuration -------------

CACHE_DIR = os.path.expanduser("~/experiment_data/datasets/ms_marco")
TOKENS_FILE = os.path.join(CACHE_DIR, "tokens.jsonl")
EMBEDS_FILE = os.path.join(CACHE_DIR, "embeddings.npy")

# Dummy word vectors (simulate pretrained embeddings like GloVe)
class DummyW2V:
    def __init__(self, dim=50):
        self.vocab = {f"word{i}": np.random.randn(dim) for i in range(10000)}
        self.dim = dim
    def __contains__(self, word): return word in self.vocab
    def __getitem__(self, word): return self.vocab.get(word, np.zeros(self.dim))

# -----------------------------------------

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def load_or_tokenize_dataset():
    if os.path.exists(TOKENS_FILE):
        print("Loading tokenized dataset...")
        with open(TOKENS_FILE, "r") as f:
            return [json.loads(line) for line in f]

    print("Tokenizing and saving dataset...")
    ds = load_dataset(
        "microsoft/ms_marco",
        "v1.1",
        cache_dir=CACHE_DIR,
        split="train"
    )

    rows = []
    for entry in tqdm(ds):
        q_tokens = tokenize(entry["query"])
        pos_tokens = [tokenize(txt) for txt in entry["passages"]["passage_text"]]
        rows.append({"query": q_tokens, "positives": pos_tokens})

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(TOKENS_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows

def compute_or_load_embeddings(tokenized_data, w2v):
    if os.path.exists(EMBEDS_FILE):
        print("Loading precomputed embeddings...")
        return np.load(EMBEDS_FILE, allow_pickle=True)

    print("Computing embeddings...")
    def embed(tokens):
        vecs = [w2v[w] for w in tokens if w in w2v]
        return np.mean(vecs, axis=0) if vecs else np.zeros(w2v.dim)

    all_embeds = []
    for entry in tqdm(tokenized_data):
        query_emb = embed(entry["query"])
        pos_embs = [embed(p) for p in entry["positives"]]
        all_embeds.append({
            "query": query_emb,
            "positives": pos_embs
        })

    np.save(EMBEDS_FILE, all_embeds)
    return all_embeds

class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.flat_negatives = []
        for entry in data:
            self.flat_negatives.extend(entry["positives"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        q = torch.tensor(item["query"], dtype=torch.float32)
        pos = torch.tensor(random.choice(item["positives"]), dtype=torch.float32)
        neg = pos
        while torch.equal(neg, pos):
            neg = torch.tensor(random.choice(self.flat_negatives), dtype=torch.float32)
        return {"query": q, "pos": pos, "neg": neg}

class AvgW2VEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=50):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, 128)

    def forward(self, batch):
        q = self.linear(batch["query"])
        pos = self.linear(batch["pos"])
        neg = self.linear(batch["neg"])
        return q, pos, neg

def train_step(model, batch):
    q, pos, neg = model(batch)
    pos_sim = torch.nn.functional.cosine_similarity(q, pos)
    neg_sim = torch.nn.functional.cosine_similarity(q, neg)
    loss = torch.relu(1 - pos_sim + neg_sim).mean()
    return loss

def main():
    # Step 1: Load dataset and tokenize
    tokenized_data = load_or_tokenize_dataset()

    # Step 2: Load dummy word2vec (replace with real one for better quality)
    w2v = DummyW2V(dim=50)

    # Step 3: Compute embeddings
    embedded_data = compute_or_load_embeddings(tokenized_data, w2v)

    # Step 4: Create dataset and model
    dataset = TripletDataset(embedded_data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = AvgW2VEncoder(embedding_dim=w2v.dim)

    # Step 5: Run a sample training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for batch in dataloader:
        loss = train_step(model, batch)
        print("Loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # Only 1 step for demo

if __name__ == "__main__":
    import cProfile
    cProfile.run("main()", sort="cumtime")

