import torch
import wandb
import argparse
from datetime import datetime
import os

from model import ImageCaptioningModel
from data import create_flickr_dataloaders
from train import train_loop
from utils import get_device, seed_all
from config import load_config, override_config_with_wandb, extract_sweep_config


def main_internal(cfg):
    seed_all(cfg.train.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader = create_flickr_dataloaders(
        device,
        cfg.dataset.cache_dir,
        cfg.dataset.valid_fraction,
        cfg.dataset.batch_size,
        cfg.dataset.num_workers)

    # TODO Expose hyperparameters of the model
    model = ImageCaptioningModel(
        decoder_type=cfg.model.decoder)

    train_loop(
        train_loader,
        valid_loader,
        device,
        model,
        cfg.train.epochs,
        cfg.train.lr,
        cfg.train.log_every,
        cfg.model.save_path_base,
        log_wandb=cfg.log.wandb)

    # Save the trained model locally
    if "save_path_base" in cfg.model:
        run_id = wandb.run.id if cfg.log.wandb else "local"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = cfg.model.save_path_base
        dynamic_path = f"{base}_{timestamp}_{run_id}_final.pt"

        os.makedirs(os.path.dirname(dynamic_path), exist_ok=True)
        torch.save(model.state_dict(), dynamic_path)
        print(f"Model saved to {dynamic_path}")

        # Optionally upload to W&B
        if cfg.log.wandb:
            artifact = wandb.Artifact("trained-model", type="model")
            artifact.add_file(dynamic_path)
            wandb.log_artifact(artifact)


def main_with_wandb(base_cfg):
    if base_cfg.log.wandb:
        is_sweep = "WANDB_SWEEP_ID" in os.environ
        wandb.init(
            project=None if is_sweep else base_cfg.log.project,
            name=base_cfg.log.run_name,
            config=extract_sweep_config(base_cfg)
        )
        cfg = override_config_with_wandb(base_cfg, wandb.config)
    else:
        cfg = base_cfg

    main_internal(cfg)

    if cfg.log.wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main_with_wandb(cfg)
