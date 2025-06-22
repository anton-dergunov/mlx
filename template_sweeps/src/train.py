import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import wandb
from model import build_model
from data import get_datasets


def train(cfg):
    print("Starting training...")
    torch.manual_seed(cfg["train"]["seed"])

    train_ds, val_ds = get_datasets(input_dim=10, dataset_size=500)
    train_loader = DataLoader(train_ds, batch_size=cfg["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["dataset"]["batch_size"])

    model = build_model(cfg["model"]["architecture"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.MSELoss()

    wandb.watch(model, log_freq=10)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch)
                val_preds.append(pred)
                val_labels.append(y_batch)

        val_preds = torch.cat(val_preds).squeeze().numpy()
        val_labels = torch.cat(val_labels).squeeze().numpy()

        val_loss = mean_absolute_error(val_labels, val_preds)
        val_r2 = r2_score(val_labels, val_preds)

        print(f"Epoch {epoch+1} | Train Loss: {sum(train_losses)/len(train_losses):.4f} | Val MAE: {val_loss:.4f} | Val R2: {val_r2:.4f}")

        if cfg["log"]["wandb"]:
            wandb.log({
                "epoch": epoch,
                "train_loss": sum(train_losses)/len(train_losses),
                "val_mae": val_loss,
                "val_r2": val_r2
            })
