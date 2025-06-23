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

    if cfg.log.wandb:
        wandb.init(
            project=cfg.log.project,
            name=cfg.log.run_name,
            config=OmegaConf.to_container(cfg, resolve=True))

    train_loader, val_loader, test_loader = load_mnist_dataloaders(
        cfg.dataset.cache_dir,
        cfg.dataset.batch_size,
        cfg.dataset.valid_fraction,
        cfg.dataset.patch_size,
        cfg.train.seed)

    model = VisionTransformer(
        cfg.dataset.patch_size * cfg.dataset.patch_size,
        cfg.model.embed_dim,
        cfg.model.num_heads,
        cfg.model.mlp_dim,
        cfg.model.num_transformer_layers,
        cfg.dataset.num_classes,
        cfg.dataset.num_patches)

    train_model(
        train_loader,
        val_loader,
        test_loader,
        device,
        model,
        cfg.train.epochs,
        cfg.train.lr)

    if cfg.log.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
