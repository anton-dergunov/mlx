from torch import nn

# -----------------------------------------------------------
# ✅ Classifiers
# -----------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TinyTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls = nn.Linear(input_dim, num_classes)

    def forward(self, x, lengths):
        # x: [batch, seq_len, dim] → Transformer expects [seq_len, batch, dim]
        x = x.transpose(0, 1)
        out = self.transformer(x)  # [seq_len, batch, dim]
        out = out.mean(dim=0)      # simple mean pooling
        return self.cls(out)
