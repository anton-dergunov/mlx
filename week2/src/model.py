import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode_query(self, x):
        pass

    @abstractmethod
    def encode_doc(self, x):
        pass

    @property
    def requires_training(self):
        return True


class BaseAvgW2VEncoder(BaseEncoder):
    def __init__(self, pad_token_idx, embedding_weights, freeze_embeddings=True):
        super().__init__()
        self.pad_token_idx = pad_token_idx
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights, freeze=freeze_embeddings, padding_idx=pad_token_idx
        )

    def _get_average_embedding(self, x):
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        mask = x != self.pad_token_idx
        emb_sum = (emb * mask.unsqueeze(-1)).sum(dim=1)
        length = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        return emb_sum / length


class AvgW2VEncoder(BaseAvgW2VEncoder):
    def __init__(self, pad_token_idx, embedding_weights, projection_dim=128, freeze_embeddings=True):
        super().__init__(pad_token_idx, embedding_weights, freeze_embeddings)
        self.query_projection = nn.Linear(embedding_weights.size(1), projection_dim)
        self.doc_projection = nn.Linear(embedding_weights.size(1), projection_dim)

    def encode_query(self, x):
        return self.query_projection(self._get_average_embedding(x))

    def encode_doc(self, x):
        return self.doc_projection(self._get_average_embedding(x))

    def forward(self, query_input, pos_input, neg_input):
        q = self.encode_query(query_input)
        pos = self.encode_doc(pos_input)
        neg = self.encode_doc(neg_input)
        return q, pos, neg


class AvgW2VEncoderNoProj(BaseAvgW2VEncoder):
    def __init__(self, pad_token_idx, embedding_weights, freeze_embeddings=True):
        super().__init__(pad_token_idx, embedding_weights, freeze_embeddings)

    def encode_query(self, x):
        return self._get_average_embedding(x)

    def encode_doc(self, x):
        return self._get_average_embedding(x)

    def forward(self, query_input, pos_input=None, neg_input=None):
        return self.encode_query(query_input)

    @property
    def requires_training(self):
        return False


def get_model(cfg, embedding_weights, pad_token_idx):
    model_type = cfg.model.type
    if model_type == "avg_w2v_encoder":
        return AvgW2VEncoder(pad_token_idx, embedding_weights, cfg.model.hidden_dim)
    elif model_type == "avg_w2v_encoder_noproj":
        return AvgW2VEncoderNoProj(pad_token_idx, embedding_weights)
    # TODO Also introduce a model that shares the weights for doc and query to test it

    raise ValueError(f"Unknown model type: {model_type}")
