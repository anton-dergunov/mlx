import argparse
import yaml
import wandb
from utils import load_config, override_config_with_wandb
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    if base_cfg["log"]["wandb"]:
        wandb.init(
            project=base_cfg["log"]["project"],
            name=base_cfg["log"]["run_name"],
            config={
                "train.lr": base_cfg["train"]["lr"],
                "train.epochs": base_cfg["train"]["epochs"],
                "model.type": base_cfg["model"]["type"]
            }
        )
        cfg = override_config_with_wandb(base_cfg, wandb.config)
    else:
        cfg = base_cfg

    train(cfg)
