import torch.nn as nn
import torch
import numpy as np
import re


class AvgW2VEncoder(torch.nn.Module):
    def __init__(self, pad_token_idx, embedding_weights, projection_dim=128, freeze_embeddings=True):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embeddings, padding_idx=pad_token_idx)  # TODO Why pass padding_idx here?
        self.projection = nn.Linear(embedding_weights.size(1), projection_dim)

    def forward(self, x):
        emb = self.embedding(x)              # (batch, seq_len, emb_dim)
        mask = x != self.pad_token_idx       # (batch, seq_len)
        emb_sum = (emb * mask.unsqueeze(-1)).sum(dim=1)    # TODO Understand this code
        length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        emb_avg = emb_sum / length
        return self.projection(emb_avg)


def get_model(cfg, embedding_weights, pad_token_idx):
    if cfg.model.type == "avg_w2v_encoder":
        return AvgW2VEncoder(pad_token_idx, embedding_weights, cfg.model.hidden_dim)

    raise ValueError(f"Unknown model type: {cfg.model.type}")
