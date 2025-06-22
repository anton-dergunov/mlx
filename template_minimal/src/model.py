import torch.nn as nn

def get_model(cfg):
    if cfg.model.type == "simple_cnn":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, cfg.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model.hidden_dim, 10)
        )
    raise ValueError(f"Unknown model type: {cfg.model.type}")
