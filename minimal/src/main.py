import os
import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf
from train import train_loop
from data import get_dataloader
from model import get_model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(cfg):
    seed_all(cfg.train.seed)

    if cfg.log.wandb:
        wandb.init(
            project=cfg.log.project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True))

    train_loader = get_dataloader(cfg)
    model = get_model(cfg)
    train_loop(model, train_loader, cfg)

    if cfg.log.wandb:
        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
