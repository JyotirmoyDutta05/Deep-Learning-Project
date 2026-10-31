from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 0
    embed_dim: int = 128
    hidden: int = 128
    cell: str = "lstm"  # "lstm" | "gru"
    dropout: float = 0.3
    max_len: int = 400
    pad_id: int = 0
