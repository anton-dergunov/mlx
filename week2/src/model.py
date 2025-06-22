import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedEncoder(nn.Module):
    def __init__(self, pad_token_idx, embedding: nn.Embedding,
                 encoder_type="avg",
                 hidden_dims=[200, 100],
                 rnn_type="gru",
                 hidden_dim=128, num_layers=1,
                 bidirectional=True, use_mean_pooling=True,
                 normalize=True):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.embedding = embedding
        self.encoder_type = encoder_type
        self.normalize = normalize

        if encoder_type == "avg_no_proj":
            self.output_dim = embedding.embedding_dim
        elif encoder_type == "avg":
            input_dim = embedding.embedding_dim
            self.encoder = self._make_projection(input_dim, hidden_dims)
            self.output_dim = hidden_dims[-1]
        elif encoder_type == "rnn":
            rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=embedding.embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.use_mean_pooling = use_mean_pooling
            self.output_dim = hidden_dim * (2 if bidirectional else 1)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _make_projection(self, input_dim, dims):
        layers = []
        full_dims = [input_dim] + dims
        for in_dim, out_dim in zip(full_dims[:-1], full_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers[:-1])

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != self.pad_token_idx)
        lengths = mask.sum(dim=1).cpu()

        if self.encoder_type in ["avg", "avg_no_proj"]:
            emb_sum = (emb * mask.unsqueeze(-1)).sum(dim=1)
            length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
            out = emb_sum / length
            if self.encoder_type == "avg":
                out = self.encoder(out)
        else:  # RNN
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnn(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            if self.use_mean_pooling:
                output = output * mask.unsqueeze(-1)
                emb_sum = output.sum(dim=1)
                length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
                out = emb_sum / length
            else:
                if isinstance(self.rnn, nn.LSTM):
                    hidden = hidden[0]
                out = hidden[-1] if self.rnn.num_layers == 1 else torch.cat([hidden[-2], hidden[-1]], dim=1)

        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    @property
    def requires_training(self):
        # TODO Create a more specific check
        return self.encoder_type != "avg_no_proj"


class DualEncoder(nn.Module):
    def __init__(self, query_encoder: UnifiedEncoder, doc_encoder: UnifiedEncoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def forward(self, query, pos_doc, neg_doc):
        q = self.query_encoder(query)
        dp = self.doc_encoder(pos_doc)
        dn = self.doc_encoder(neg_doc)
        return q, dp, dn

    def encode_query(self, query):
        return self.query_encoder(query)

    def encode_document(self, query):
        return self.doc_encoder(query)

    @property
    def requires_training(self):
        return self.query_encoder.requires_training or self.doc_encoder.requires_training


class CrossEncoder(nn.Module):
    def __init__(self, encoder: UnifiedEncoder):
        super().__init__()
        self.encoder = encoder
        self.scorer = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, query, document):
        x = torch.cat([query, document], dim=1)
        enc = self.encoder(x)
        score = self.scorer(enc)
        return score.squeeze(-1)

    @property
    def requires_training(self):
        return True


def build_model(cfg_model, embedding: nn.Embedding, pad_token_idx):
    if cfg_model.type == "dual_encoder":
        query_encoder = UnifiedEncoder(
            pad_token_idx=pad_token_idx,
            embedding=embedding if cfg_model.shared_embedding else embedding.clone(),
            **cfg_model.dual_encoder.query
        )
        doc_encoder = UnifiedEncoder(
            pad_token_idx=pad_token_idx,
            embedding=embedding if cfg_model.shared_embedding else embedding.clone(),
            **cfg_model.dual_encoder.document
        )
        return DualEncoder(query_encoder, doc_encoder)

    elif cfg_model.type == "cross_encoder":
        encoder = UnifiedEncoder(
            pad_token_idx=pad_token_idx,
            embedding=embedding,
            **cfg_model.cross_encoder
        )
        return CrossEncoder(encoder)

    else:
        raise ValueError(f"Unknown model type: {cfg_model.type}")
    

def create_shared_embedding(embedding_weights, pad_token_idx, freeze=True):
    return nn.Embedding.from_pretrained(embedding_weights, freeze=freeze, padding_idx=pad_token_idx)


def save_model(model: nn.Module, path: str):
    state_dict = model.state_dict()

    # Remove embedding if it's frozen
    keys_to_remove = [k for k, v in model.named_parameters() if not v.requires_grad and "embedding" in k]
    for key in keys_to_remove:
        state_dict.pop(key, None)

    torch.save({
        "model_state_dict": state_dict,
    }, path)


# TODO What is strict?
def load_model(model: nn.Module, path: str, device, strict=False):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
