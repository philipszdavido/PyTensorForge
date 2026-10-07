import logging
import threading
import time
from dataclasses import dataclass

from pytensorforge.inference.runtime import load_model
from pytensorforge.serving import errors
from pytensorforge.models.gpt.context import describe_context

log = logging.getLogger("ptf.server")

SUPPORTED_DEVICES = ("cpu",)


def resolve_device(requested):
    requested = (requested or "auto").lower()

    if requested in ("auto", "cpu"):
        return "cpu"

    raise ValueError(
        f"device '{requested}' is not available: this build's inference engine runs on {SUPPORTED_DEVICES} only"
    )


@dataclass
class ModelLimits:
    context_length: int
    max_generation_tokens: int
    max_prompt_tokens: int


class ServedModel:

    def __init__(self, entry, generator, template, template_source, limits, device):
        self.entry = entry
        self.name = entry.name
        self.generator = generator
        self.template = template
        self.template_source = template_source
        self.limits = limits
        self.device = device
        self.loaded_at = int(time.time())
        self.in_flight = 0
        self.accepting = True
        self._idle = threading.Condition()

    @property
    def fingerprint(self):
        return "ptf-" + self.generator.metadata.get("weights_sha256", "")[:12]

    @property
    def cache_budget_bytes(self):
        return self.generator.scheduler.cache_manager.max_bytes

    @property
    def memory_bytes(self):
        return self.generator.weights_nbytes + (self.cache_budget_bytes or 0)

    def enter(self):
        with self._idle:
            if not self.accepting:
                raise errors.overloaded(f"model '{self.name}' is being unloaded")

            self.in_flight += 1

    def exit(self):
        with self._idle:
            self.in_flight -= 1

            if self.in_flight <= 0:
                self._idle.notify_all()

    def wait_idle(self, timeout):
        deadline = time.monotonic() + timeout

        with self._idle:
            while self.in_flight > 0:
                left = deadline - time.monotonic()

                if left <= 0:
                    return False

                self._idle.wait(left)

        return True

    def describe(self):
        cfg = self.generator.config

        return {
            "id": self.name,
            "object": "model",
            "created": self.loaded_at,
            "owned_by": "pytensorforge",
            "context_length": self.limits.context_length,
            "context": describe_context(cfg),
            "max_generation_tokens": self.limits.max_generation_tokens,
            "chat_template": self.template.name,
            "chat_template_source": self.template_source,
            "architecture": cfg.architecture,
            "vocab_size": cfg.vocab_size,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "device": self.device,
            "fingerprint": self.fingerprint,
        }


class ModelServer:

    def __init__(self, config):
        self.config = config
        self.device = resolve_device(config.runtime.device)
        self._entries = {m.name: m for m in config.models}
        self._models = {}
        self._lock = threading.RLock()
        self._loading = set()
        self._global_in_flight = 0
        self._global_lock = threading.Lock()

    def declared(self):
        return list(self._entries.values())

    def loaded_models(self):
        with self._lock:
            return list(self._models.values())

    def get(self, name):
        with self._lock:
            model = self._models.get(name)

        if model is None or not model.accepting:
            if name in self._entries:
                raise errors.APIError(503, f"model '{name}' is not loaded", "server_error", code="model_not_loaded")

            raise errors.not_found(f"the model '{name}' does not exist", code="model_not_found")

        return model

    def _default_budget(self, generator, entry):
        ctx = generator.config.context_length

        if entry.max_context_tokens:
            ctx = min(ctx, entry.max_context_tokens)

        return generator.scheduler.cache_manager.bytes_for(ctx) * entry.max_batch_size

    def _committed_bytes(self, exclude=None):
        return sum(m.memory_bytes for n, m in self._models.items() if n != exclude)

    def load(self, name):
        entry = self._entries.get(name)

        if entry is None:
            raise errors.not_found(f"model '{name}' is not declared in the server configuration", code="model_not_found")

        with self._lock:
            if name in self._models:
                return self._models[name]

            if name in self._loading:
                raise errors.APIError(409, f"model '{name}' is already loading", "invalid_request_error", code="conflict")

            self._loading.add(name)

        try:
            return self._load(entry)
        finally:
            with self._lock:
                self._loading.discard(name)

    def _load(self, entry):
        started = time.perf_counter()
        budget = int(entry.kv_cache_budget_mb * (1 << 20)) if entry.kv_cache_budget_mb else None

        generator = load_model(
            entry.path,
            max_batch_size=entry.max_batch_size,
            cache_budget_bytes=budget,
            prefill_chunk=entry.prefill_chunk,
            chat_template=entry.chat_template,
            context_length=entry.extend_context_to,
            context_extension=entry.context_extension,
        )

        if budget is None:
            generator.scheduler.cache_manager.max_bytes = self._default_budget(generator, entry)

        if generator.chat_template is not None:
            source = "config" if entry.chat_template else "model"
            template = generator.bound_chat_template()
        else:
            source = "fallback"
            template = generator.bound_chat_template(self.config.runtime.fallback_chat_template)
            log.warning("model '%s' has no chat template; using fallback '%s'", entry.name, template.name)

        ctx = generator.config.context_length

        if entry.max_context_tokens:
            ctx = min(ctx, entry.max_context_tokens)

        max_gen = min(entry.max_generation_tokens or self.config.limits.max_generation_tokens, ctx - 1)
        max_prompt = min(self.config.limits.max_prompt_tokens or ctx, ctx - 1)
        limits = ModelLimits(context_length=ctx, max_generation_tokens=max_gen, max_prompt_tokens=max_prompt)

        model = ServedModel(entry, generator, template, source, limits, self.device)

        with self._lock:
            limit_mb = self.config.runtime.memory_limit_mb

            if limit_mb is not None:
                needed = self._committed_bytes() + model.memory_bytes

                if needed > limit_mb * (1 << 20):
                    raise errors.APIError(
                        507,
                        f"loading '{entry.name}' needs {model.memory_bytes / (1 << 20):.1f} MiB "
                        f"(weights + KV cache budget); memory limit of {limit_mb:.0f} MiB would be exceeded",
                        "server_error",
                        code="insufficient_memory",
                    )

            generator.start()
            self._models[entry.name] = model

        log.info("loaded model '%s' in %.2fs (%.1f MiB weights, %.1f MiB kv budget, template %s/%s)",
                 entry.name, time.perf_counter() - started, generator.weights_nbytes / (1 << 20),
                 (model.cache_budget_bytes or 0) / (1 << 20), template.name, source)

        return model

    def unload(self, name, drain_timeout_s=30.0):
        with self._lock:
            model = self._models.get(name)

            if model is None:
                raise errors.not_found(f"model '{name}' is not loaded", code="model_not_loaded")

            model.accepting = False

        drained = model.wait_idle(drain_timeout_s)

        if not drained:
            sched = model.generator.scheduler

            for rid in list(sched._active.keys()) + [h.request_id for h in list(sched._waiting)]:
                sched.request_cancel(rid)

            model.wait_idle(5.0)

        model.generator.stop()

        with self._lock:
            self._models.pop(name, None)

        log.info("unloaded model '%s' (%s)", name, "drained" if drained else "cancelled in-flight requests")

        return drained

    def preload(self):
        for entry in self._entries.values():
            if entry.preload:
                self.load(entry.name)

    def acquire_global(self):
        with self._global_lock:
            if self._global_in_flight >= self.config.limits.max_concurrent_requests:
                raise errors.overloaded()

            self._global_in_flight += 1

    def release_global(self):
        with self._global_lock:
            self._global_in_flight -= 1

    @property
    def global_in_flight(self):
        return self._global_in_flight

    def shutdown(self, drain_timeout_s):
        for model in self.loaded_models():
            model.accepting = False

        for model in self.loaded_models():
            self.unload(model.name, drain_timeout_s)
