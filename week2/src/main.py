import os
import torch
import wandb
import random
import numpy as np
from omegaconf import OmegaConf

from train import train_loop
from data import get_dataloader
from model import get_model
from embeddings import get_pretrained_w2v_embeddings, PAD_TOKEN


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
            name=cfg.log.run_name,
            config=OmegaConf.to_container(cfg, resolve=True))
        
    print("Loading pretrained word2vec model...")
    embedding_matrix, word2idx = get_pretrained_w2v_embeddings(cfg)

    print("Loading dataset...")
    train_loader = get_dataloader(cfg, word2idx)

    model = get_model(cfg, embedding_matrix, word2idx[PAD_TOKEN])

    if not model.requires_training:
        print("Skip training")
    else:
        print("Starting training...")
        train_loop(model, train_loader, cfg)

    if cfg.log.wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    # TODO Load custom and base (default) configs
    cfg = OmegaConf.load(args.config)
    main(cfg)
