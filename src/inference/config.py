from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    seed: Optional[int] = None
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)
    stop_on_eos: bool = True

    def __post_init__(self):
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

    @property
    def greedy(self):
        return (not self.do_sample) or self.temperature == 0

    def updated(self, **overrides):
        overrides = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **overrides)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known})
