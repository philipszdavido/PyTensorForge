import json
import os
import re
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional

MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass
class ModelEntry:
    name: str
    path: str
    chat_template: Optional[str] = None
    max_batch_size: int = 8
    kv_cache_budget_mb: Optional[float] = None
    prefill_chunk: int = 256
    preload: bool = True
    max_context_tokens: Optional[int] = None
    max_generation_tokens: Optional[int] = None
    extend_context_to: Optional[int] = None
    context_extension: Optional[str] = None

    def validate(self):
        if (self.extend_context_to is None) != (self.context_extension is None):
            raise ValueError(f"model '{self.name}': extend_context_to and context_extension go together")

        if self.context_extension is not None and self.context_extension not in ("extrapolate", "linear", "ntk"):
            raise ValueError(f"model '{self.name}': context_extension must be extrapolate, linear or ntk")

        if not MODEL_NAME_RE.match(self.name):
            raise ValueError(f"invalid model name '{self.name}' (letters, digits, . _ : - ; max 128)")

        if self.max_batch_size < 1:
            raise ValueError(f"model '{self.name}': max_batch_size must be >= 1")

        if self.kv_cache_budget_mb is not None and self.kv_cache_budget_mb <= 0:
            raise ValueError(f"model '{self.name}': kv_cache_budget_mb must be > 0")

        if self.prefill_chunk < 1:
            raise ValueError(f"model '{self.name}': prefill_chunk must be >= 1")


@dataclass
class LimitsConfig:
    max_concurrent_requests: int = 64
    max_body_bytes: int = 1 << 20
    max_header_bytes: int = 16 << 10
    max_messages: int = 256
    max_prompt_chars: int = 200_000
    max_prompt_tokens: Optional[int] = None
    max_generation_tokens: int = 1024
    max_stop_sequences: int = 4
    max_stop_sequence_chars: int = 128
    request_timeout_s: float = 300.0
    max_connections: int = 512
    header_timeout_s: float = 10.0
    body_timeout_s: float = 30.0
    keepalive_timeout_s: float = 30.0
    max_requests_per_connection: int = 1000
    sse_keepalive_s: float = 15.0
    shutdown_drain_s: float = 30.0

    def validate(self):
        for name in ("max_concurrent_requests", "max_body_bytes", "max_header_bytes", "max_messages",
                     "max_prompt_chars", "max_generation_tokens", "max_connections"):
            if getattr(self, name) < 1:
                raise ValueError(f"limits.{name} must be >= 1")

        for name in ("request_timeout_s", "header_timeout_s", "body_timeout_s", "keepalive_timeout_s", "sse_keepalive_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"limits.{name} must be > 0")


@dataclass
class RateLimitConfig:
    requests_per_minute: float = 120.0
    burst: int = 30
    max_concurrent_per_principal: int = 8
    max_tracked_principals: int = 10_000

    def validate(self):
        if self.requests_per_minute <= 0 or self.burst < 1 or self.max_concurrent_per_principal < 1:
            raise ValueError("rate_limit values must be positive")


@dataclass
class SecurityConfig:
    api_keys: List[str] = field(default_factory=list)
    api_keys_file: Optional[str] = None
    api_keys_env: Optional[str] = "PTF_API_KEYS"
    admin_keys: List[str] = field(default_factory=list)
    admin_keys_file: Optional[str] = None
    allow_unauthenticated: bool = False
    metrics_require_auth: bool = True
    cors_origins: List[str] = field(default_factory=list)
    trust_forwarded_for: bool = False
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)


@dataclass
class RuntimeConfig:
    device: str = "auto"
    memory_limit_mb: Optional[float] = None
    fallback_chat_template: str = "plain"
    chat_truncation: str = "auto"


@dataclass
class LoggingConfig:
    access_log: Optional[str] = "-"
    log_prompts: bool = False


@dataclass
class UIConfig:
    enabled: bool = True
    directory: Optional[str] = None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    models: List[ModelEntry] = field(default_factory=list)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def validate(self):
        if not self.models:
            raise ValueError("at least one model must be configured")

        names = [m.name for m in self.models]

        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")

        for m in self.models:
            m.validate()

        if not 0 <= self.port < 65536:
            raise ValueError("port must be in 0..65535 (0 picks a free port)")

        if self.runtime.chat_truncation not in ("auto", "disabled"):
            raise ValueError("runtime.chat_truncation must be 'auto' or 'disabled'")

        self.limits.validate()
        self.security.rate_limit.validate()

        return self

    def to_dict(self):
        return asdict(self)


def _build(cls, data, where):
    if data is None:
        return cls()

    if not isinstance(data, dict):
        raise ValueError(f"'{where}' must be a mapping")

    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))

    if unknown:
        raise ValueError(f"unknown keys in '{where}': {unknown}")

    kwargs = {}

    for name, value in data.items():
        f = known[name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default

        if is_dataclass(default) and not isinstance(default, type):
            kwargs[name] = _build(type(default), value, f"{where}.{name}")
        else:
            kwargs[name] = value

    return cls(**kwargs)


def server_config_from_dict(data):
    data = dict(data or {})
    models = data.pop("models", [])

    if not isinstance(models, list):
        raise ValueError("'models' must be a list")

    cfg = _build(ServerConfig, data, "server")
    cfg.models = [_build(ModelEntry, m, f"models[{i}]") for i, m in enumerate(models)]

    return cfg


def load_server_config(path):
    with open(path) as f:
        text = f.read()

    if path.endswith((".yaml", ".yml")):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    cfg = server_config_from_dict(data)
    base = os.path.dirname(os.path.abspath(path))

    for m in cfg.models:
        if not os.path.isabs(m.path):
            m.path = os.path.join(base, m.path)

    return cfg
