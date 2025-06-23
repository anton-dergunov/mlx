import torch
import wandb
from omegaconf import OmegaConf
import random
import numpy as np
import argparse

from model import VisionTransformer
from data import load_mnist_dataloaders
from train import train_model
from utils import get_device
from config import load_config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg):
    seed_all(cfg.train.seed)

    device = get_device()
    print(f"Using device: {device}")

    # if cfg.log.wandb:
    wandb.init(
        project=cfg.log.project,
        name=cfg.log.run_name,
        config=OmegaConf.to_container(cfg, resolve=True))

    cache_dir = "~/experiment_data/datasets"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_mnist_dataloaders(cache_dir, batch_size=64)

    model = VisionTransformer()

    train_model(train_loader, val_loader, test_loader, device, model, num_epochs=10)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
