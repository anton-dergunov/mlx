import yaml
import copy

SWEEPABLE_KEYS = {
    "train.lr",
    "train.epochs",
    "model.architecture"
}

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def extract_sweep_config(cfg):
    result = {}
    for key in SWEEPABLE_KEYS:
        sub_cfg = cfg
        keys = key.split(".")
        try:
            for k in keys:
                sub_cfg = sub_cfg[k]
            result[key] = sub_cfg
        except (KeyError, TypeError):
            continue
    return result

def override_config_with_wandb(base_cfg, wandb_cfg):
    cfg = copy.deepcopy(base_cfg)
    for key, value in wandb_cfg.items():
        if key not in SWEEPABLE_KEYS:
            continue
        if "." in key:
            keys = key.split(".")
            sub_cfg = cfg
            for k in keys[:-1]:
                sub_cfg = sub_cfg.setdefault(k, {})
            sub_cfg[keys[-1]] = value
        else:
            cfg[key] = value
    return cfg
