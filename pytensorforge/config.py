from dataclasses import asdict, dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ModelSpec:
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
    position_encoding: str = "learned"
    rope_theta: float = 10000.0
    rope_scaling: str = "none"
    rope_scaling_factor: float = 1.0
    trained_context_length: Optional[int] = None


@dataclass
class DataSpec:
    train: List[str] = field(default_factory=list)
    validation: Optional[List[str]] = None
    tokenizer: str = "tokenizer.json"
    sequence_length: int = 1024
    read_buffer_size: int = 1 << 20
    insert_eos: bool = True
    text_field: str = "text"
    workers: int = 1
    prefetch: int = 4
    shuffle_files: bool = False
    val_split_fraction: Optional[float] = None
    format: str = "text"
    chat_template: str = "plain"
    chat_packing: str = "pack"
    messages_field: str = "messages"


@dataclass
class TrainingSpec:
    max_tokens: Optional[int] = None
    max_epochs: Optional[int] = None
    total_steps: Optional[int] = None
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    decay: str = "cosine"
    min_lr: float = 0.0
    gradient_clip: float = 1.0
    precision: str = "fp32"
    loss_scaling: str = "auto"
    initial_loss_scale: float = 65536.0
    loss_scale_growth_interval: int = 2000
    activation_checkpointing: bool = False
    init_from: Optional[str] = None
    seed: int = 1337


@dataclass
class CheckpointSpec:
    directory: str = "checkpoints"
    interval_steps: int = 500
    keep_last: int = 3
    keep_every: Optional[int] = None


@dataclass
class EvaluationSpec:
    interval_steps: int = 500
    interval_tokens: Optional[int] = None
    max_batches: int = 50


@dataclass
class RuntimeSpec:
    device: str = "cpu"
    seed: int = 1337


@dataclass
class TrainConfig:
    model: ModelSpec
    data: DataSpec
    training: TrainingSpec
    checkpoint: CheckpointSpec
    evaluation: EvaluationSpec
    runtime: RuntimeSpec

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        return cls(
            model=ModelSpec(**raw.get("model", {})),
            data=DataSpec(**raw.get("data", {})),
            training=TrainingSpec(**raw.get("training", {})),
            checkpoint=CheckpointSpec(**raw.get("checkpoint", {})),
            evaluation=EvaluationSpec(**raw.get("evaluation", {})),
            runtime=RuntimeSpec(**raw.get("runtime", {})),
        )

    @classmethod
    def load(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls.from_dict(raw)

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
