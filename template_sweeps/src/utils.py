import yaml
import copy

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def override_config_with_wandb(base_cfg, wandb_cfg):
    cfg = copy.deepcopy(base_cfg)
    for key, value in wandb_cfg.items():
        if "." in key:
            keys = key.split(".")
            sub_cfg = cfg
            for k in keys[:-1]:
                sub_cfg = sub_cfg.setdefault(k, {})
            sub_cfg[keys[-1]] = value
        else:
            cfg[key] = value
    return cfg
