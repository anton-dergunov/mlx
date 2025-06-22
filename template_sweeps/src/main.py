import argparse
import wandb
import os

from config import load_config, override_config_with_wandb, extract_sweep_config
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    if base_cfg["log"]["wandb"]:
        is_sweep = "WANDB_SWEEP_ID" in os.environ
        wandb.init(
            project=None if is_sweep else base_cfg["log"]["project"],
            name=base_cfg["log"]["run_name"],
            config=extract_sweep_config(base_cfg)
        )
        cfg = override_config_with_wandb(base_cfg, wandb.config)
    else:
        cfg = base_cfg

    train(cfg)

    if cfg["log"]["wandb"]:
        wandb.finish()
