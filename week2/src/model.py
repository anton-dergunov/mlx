import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def requires_training(self):
        return True


class BaseAvgW2VEncoder(BaseEncoder):
    def __init__(self, pad_token_idx, embedding: nn.Embedding):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.embedding = embedding

    def _get_average_embedding(self, x):
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        mask = x != self.pad_token_idx
        emb_sum = (emb * mask.unsqueeze(-1)).sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        return emb_sum / length


class AvgW2VEncoder(BaseAvgW2VEncoder):
    def __init__(self, pad_token_idx, embedding: nn.Embedding, projection_dims=[200, 100]):
        super().__init__(pad_token_idx, embedding)
        input_dim = embedding.embedding_dim
        self.projection = self._make_projection(input_dim, projection_dims)

    def _make_projection(self, input_dim, projection_dims):
        layers = []
        dims = [input_dim] + projection_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        # Remove last ReLU for output layer
        if layers:
            layers = layers[:-1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.projection(self._get_average_embedding(x))
        return F.normalize(out, p=2, dim=1)


class AvgW2VEncoderNoProj(BaseAvgW2VEncoder):
    def __init__(self, pad_token_idx, embedding: nn.Embedding):
        super().__init__(pad_token_idx, embedding)

    def forward(self, x):
        return self._get_average_embedding(x)

    @property
    def requires_training(self):
        return False


class BiGRUEncoder(BaseAvgW2VEncoder):
    def __init__(self, pad_token_idx, embedding: nn.Embedding, hidden_dim=128, num_layers=1):
        super().__init__(pad_token_idx, embedding)
        self.gru = nn.GRU(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        mask = (x != self.pad_token_idx)
        lengths = mask.sum(dim=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Mean pooling over time (with mask)
        output = output * mask.unsqueeze(-1)
        emb_sum = output.sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        out = emb_sum / length
        return F.normalize(out, p=2, dim=1)


def create_shared_embedding(embedding_weights, pad_token_idx, freeze=True):
    return nn.Embedding.from_pretrained(embedding_weights, freeze=freeze, padding_idx=pad_token_idx)


def get_model(cfg_model, embedding: nn.Embedding, pad_token_idx):
    model_type = cfg_model.type
    if model_type == "avg_w2v_encoder":
        return AvgW2VEncoder(pad_token_idx, embedding, cfg_model.hidden_dims)
    elif model_type == "avg_w2v_encoder_noproj":
        return AvgW2VEncoderNoProj(pad_token_idx, embedding)
    elif model_type == "bi_gru_encoder":
        return BiGRUEncoder(pad_token_idx, embedding, cfg_model.hidden_dim, cfg_model.num_layers)

    # TODO Also introduce a model that shares the weights for doc and query to test it

    raise ValueError(f"Unknown model type: {model_type}")
