from dataclasses import dataclass, field
from typing import List, Optional, Literal
from omegaconf import MISSING, OmegaConf


# TODO: Seems to require Python 3.12
# EncoderType = Literal["avg", "rnn", "avg_no_proj"]
# RNNType = Literal["gru", "lstm"]
# ModelType = Literal["dual_encoder", "cross_encoder"]


# ---- Encoder Configs ----


@dataclass
class EncoderConfig:
    # TODO: Seems to require Python 3.12
    # encoder_type: EncoderType = "avg"
    encoder_type: str = "avg"
    hidden_dims: Optional[List[int]] = None
    # TODO: Seems to require Python 3.12
    # rnn_type: Optional[RNNType] = None
    rnn_type: Optional[str] = None
    hidden_dim: Optional[int] = None
    num_layers: int = 1
    bidirectional: bool = True
    use_mean_pooling: bool = True
    normalize: bool = True


@dataclass
class DualEncoderConfig:
    query: EncoderConfig = field(default_factory=EncoderConfig)
    document: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class CrossEncoderConfig:
    # TODO: Seems to require Python 3.12
    # encoder_type: EncoderType = "avg"
    encoder_type: str = "avg"
    hidden_dims: Optional[List[int]] = None
    # TODO: Seems to require Python 3.12
    # rnn_type: Optional[RNNType] = None
    rnn_type: Optional[str] = None
    hidden_dim: Optional[int] = None
    num_layers: int = 1
    bidirectional: bool = True
    use_mean_pooling: bool = True
    normalize: bool = True


# ---- Model ----


@dataclass
class ModelConfig:
    # TODO: Seems to require Python 3.12
    # type: ModelType = "dual_encoder"
    type: str = "dual_encoder"
    shared_embedding: bool = True
    freeze_embedding: bool = True
    dual_encoder: Optional[DualEncoderConfig] = None
    cross_encoder: Optional[CrossEncoderConfig] = None
    save_path: Optional[str] = None


# ---- Dataset ----


@dataclass
class DatasetConfig:
    name: str = "microsoft/ms_marco"
    version: str = "v1.1"
    batch_size: int = 64
    cache_dir: Optional[str] = None


# ---- Embeddings ----


@dataclass
class EmbeddingConfig:
    path: str = MISSING  # force user to specify
    is_binary: bool = True


# ---- Train ----


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42


# ---- Logging ----


@dataclass
class LoggingConfig:
    wandb: bool = False
    project: Optional[str] = None
    run_name: Optional[str] = None


# ---- Top-level config ----


@dataclass
class Config:
    project: str = "ms-marco-ranking"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(yaml_path: str) -> Config:
    schema = OmegaConf.structured(Config())
    # TODO Load custom and base (default) configs
    user_cfg = OmegaConf.load(yaml_path)
    merged = OmegaConf.merge(schema, user_cfg)

    assert merged.model.type in {"dual_encoder", "cross_encoder"}, f"Unknown model type: {merged.model.type}"

    return merged
