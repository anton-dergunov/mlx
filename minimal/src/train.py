import torch
import wandb
import torch.nn.functional as F
from torch import nn, optim

def train_loop(model, dataloader, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
        if cfg.log.wandb:
            wandb.log({"epoch": epoch, "loss": avg_loss})
