import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        if model_type == "dual_encoder":
            self.net = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:  # "cross_encoder"
            self.net = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(self, x):
        return self.net(x)
