import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
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

    @property
    def requires_training(self):
        return True


class AvgW2VEncoder(BaseEncoder):
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


class AvgW2VEncoderNoProj(BaseEncoder):
    def __init__(self, pad_token_idx, embedding: nn.Embedding):
        super().__init__(pad_token_idx, embedding)

    def forward(self, x):
        return self._get_average_embedding(x)

    @property
    def requires_training(self):
        return False


class GenericRNNEncoder(BaseEncoder):
    def __init__(self, pad_token_idx, embedding: nn.Embedding, 
                 rnn_class=nn.GRU, hidden_dim=128, num_layers=1, bidirectional=True, use_mean_pooling=True):
        super().__init__(pad_token_idx, embedding)
        self.bidirectional = bidirectional
        self.use_mean_pooling = use_mean_pooling
        self.rnn = rnn_class(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != self.pad_token_idx)
        lengths = mask.sum(dim=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        if self.use_mean_pooling:
            output = output * mask.unsqueeze(-1)
            emb_sum = output.sum(dim=1)
            length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            out = emb_sum / length
        else:
            # Use final hidden state(s)
            if isinstance(self.rnn, nn.LSTM):
                hidden = hidden[0]  # LSTM: (hidden_state, cell_state)
            out = hidden[-1] if not self.bidirectional else torch.cat([hidden[-2], hidden[-1]], dim=1)

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
        return GenericRNNEncoder(pad_token_idx, embedding, nn.GRU, cfg_model.hidden_dim, cfg_model.num_layers, bidirectional=True, use_mean_pooling=True)
    elif model_type == "bi_lstm_encoder":
        return GenericRNNEncoder(pad_token_idx, embedding, nn.LSTM, cfg_model.hidden_dim, cfg_model.num_layers, bidirectional=True, use_mean_pooling=True)
    elif model_type == "uni_gru_encoder":
        return GenericRNNEncoder(pad_token_idx, embedding, nn.GRU, cfg_model.hidden_dim, cfg_model.num_layers, bidirectional=False, use_mean_pooling=False)

    raise ValueError(f"Unknown model type: {model_type}")
