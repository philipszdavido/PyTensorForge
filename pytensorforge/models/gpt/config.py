import json
from typing import Optional
from dataclasses import asdict, dataclass


SUPPORTED_ACTIVATIONS = ("gelu", "relu", "tanh", "sigmoid")
POSITION_ENCODINGS = ("learned", "rope")


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ff_dim: int = 3072
    activation: str = "gelu"
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_weights: bool = True
    architecture: str = "gpt"
    position_encoding: str = "learned"
    rope_theta: float = 10000.0
    rope_scaling: str = "none"
    rope_scaling_factor: float = 1.0
    trained_context_length: Optional[int] = None

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported activation '{self.activation}'; choose one of {SUPPORTED_ACTIVATIONS}"
            )

        if self.position_encoding not in POSITION_ENCODINGS:
            raise ValueError(
                f"unsupported position_encoding '{self.position_encoding}'; choose one of {POSITION_ENCODINGS}"
            )

        if self.position_encoding == "rope":
            if (self.d_model // self.n_heads) % 2 != 0:
                raise ValueError("rope needs an even head dimension (d_model / n_heads)")

            if not self.rope_theta > 0:
                raise ValueError("rope_theta must be positive")

        if self.rope_scaling not in ("none", "linear", "ntk"):
            raise ValueError(f"unknown rope_scaling '{self.rope_scaling}'; choose none, linear or ntk")

        self.rope_scaling_factor = float(self.rope_scaling_factor)

        if not self.rope_scaling_factor >= 1.0:
            raise ValueError("rope_scaling_factor must be >= 1")

        if (self.rope_scaling != "none" or self.trained_context_length is not None) and not self.uses_rope:
            raise ValueError("context extension settings require position_encoding: rope")

        if self.trained_context_length is not None and self.trained_context_length > self.context_length:
            raise ValueError("trained_context_length cannot exceed context_length")

        self.rope_theta = float(self.rope_theta)

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def uses_rope(self):
        return self.position_encoding == "rope"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def base_context_length(self):
        return self.trained_context_length or self.context_length

    def rotary_tables(self):
        from pytensorforge.models.gpt.rope import rotary_tables

        return rotary_tables(self.head_dim, self.context_length, self.rope_theta,
                             self.rope_scaling, self.rope_scaling_factor)

    @classmethod
    def normalized(cls, data):
        return cls.from_dict(data).to_dict()

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
