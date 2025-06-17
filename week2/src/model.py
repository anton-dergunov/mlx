import torch.nn as nn
import torch
from gensim.models import KeyedVectors
import numpy as np
import re


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


def get_model(cfg, device):
    # Download the data from https://code.google.com/archive/p/word2vec/
    # TODO Replace with configurable path
    print("Loading pretrained word2vec model...")
    w2v = KeyedVectors.load_word2vec_format("~/experiment_data/datasets/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", binary=True)

    if cfg.model.type == "avg_w2v_encoder":
        return AvgW2VEncoder(w2v_model=w2v, hidden_dim=cfg.model.hidden_dim, device=device).to(device)

    raise ValueError(f"Unknown model type: {cfg.model.type}")
