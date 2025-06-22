import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleModel
import wandb

def train(cfg):
    print("Starting training...")
    torch.manual_seed(cfg["train"]["seed"])
    model = SimpleModel(cfg["model"]["type"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.MSELoss()

    # Dummy dataset
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg["dataset"]["batch_size"])

    wandb.watch(model, log_freq=10)

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

        if cfg["log"]["wandb"]:
            wandb.log({"epoch": epoch, "loss": loss.item()})
