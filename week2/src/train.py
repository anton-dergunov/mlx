import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import wandb

from utils import get_device


def train_loop(model, dataloader, cfg):
    device = get_device()
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            q, pos, neg = batch["query"], batch["pos"], batch["neg"]
            q_emb = model(q)
            pos_emb = model(pos)
            neg_emb = model(neg)

            loss = triplet_loss(q_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        if cfg.log.wandb:
            wandb.log({"epoch": epoch, "loss": avg_loss})
