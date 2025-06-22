import torch.nn as nn

def build_model(architecture: str, input_dim: int = 10) -> nn.Module:
    if architecture == "shallow_relu":
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    elif architecture == "deep_tanh":
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    