import json
import os

import numpy as np

from src.inference.chat_template import ChatTemplate
from src.inference.config import GenerationConfig
from src.inference.engine import InferenceModel
from src.inference.export import EXPORT_FORMAT_VERSION, _sha256
from src.inference.scheduler import BatchScheduler, GenerationRequest
from src.models.gpt.config import GPTConfig
from src.models.gpt.context import describe_context, extend_context
from src.tokenization.base import identity_matches
from src.tokenization.registry import load_tokenizer


class TextGenerator:

    def __init__(
        self,
        engine,
        tokenizer,
        default_config=None,
        max_batch_size=8,
        cache_budget_bytes=None,
        prefill_chunk=256,
        chat_template=None,
        metadata=None,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.default_config = default_config or GenerationConfig()
        self.chat_template = chat_template
        self.metadata = metadata or {}
        self.scheduler = BatchScheduler(
            engine,
            tokenizer,
            max_batch_size=max_batch_size,
            cache_budget_bytes=cache_budget_bytes,
            prefill_chunk=prefill_chunk,
        )

    @property
    def config(self):
        return self.engine.config

    @property
    def weights_nbytes(self):
        return self.engine.nbytes

    def bound_chat_template(self, fallback=None):
        template = self.chat_template or ChatTemplate.resolve(fallback)

        if template is None:
            raise ValueError("model has no chat template and no fallback was given")

        return template.bind(self.tokenizer)

    def _request(self, prompt, overrides, timeout_s=None, truncate_prompt=False, request_id=None):
        kwargs = {}

        if request_id:
            kwargs["request_id"] = request_id

        return GenerationRequest(
            prompt=prompt,
            config=self.default_config.updated(**overrides),
            timeout_s=timeout_s,
            truncate_prompt=truncate_prompt,
            **kwargs,
        )

    def submit(self, request):
        return self.scheduler.submit(request)

    def generate(self, prompt, timeout_s=None, truncate_prompt=False, **overrides):
        handle = self.submit(self._request(prompt, overrides, timeout_s, truncate_prompt))
        return handle.result(drive=not self.scheduler.background)

    def generate_stream(self, prompt, timeout_s=None, truncate_prompt=False, events=False, **overrides):
        handle = self.submit(self._request(prompt, overrides, timeout_s, truncate_prompt))

        stream = handle.events(drive=not self.scheduler.background)

        try:
            for ev in stream:
                if events:
                    yield ev
                elif ev.text:
                    yield ev.text
        finally:
            stream.close()
            handle.cancel()

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.stop()


def load_model(path, max_batch_size=8, cache_budget_bytes=None, verify=True, prefill_chunk=256, chat_template=None,
               context_length=None, context_extension=None):
    with open(os.path.join(path, "config.json")) as f:
        meta = json.load(f)

    if meta["format_version"] != EXPORT_FORMAT_VERSION:
        raise ValueError(f"unsupported export format version {meta['format_version']}")

    weights_path = os.path.join(path, meta["weights"]["file"])

    if verify and _sha256(weights_path) != meta["weights"]["sha256"]:
        raise ValueError("model weights failed checksum verification")

    config = GPTConfig.from_dict(meta["model"])

    if context_length is not None or context_extension is not None:
        if context_length is None or context_extension is None:
            raise ValueError("context extension needs both context_length and context_extension")

        config = extend_context(config, context_length, context_extension)

    with np.load(weights_path, allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}

    tokenizer = load_tokenizer(os.path.join(path, meta["tokenizer"]["file"]))

    if not identity_matches(meta["tokenizer"]["identity"], tokenizer.identity):
        raise ValueError("bundled tokenizer does not match the exported model")

    gen_path = os.path.join(path, "generation.json")
    default = GenerationConfig()

    if os.path.isfile(gen_path):
        with open(gen_path) as f:
            default = GenerationConfig.from_dict(json.load(f))

    template = ChatTemplate.resolve(chat_template if chat_template is not None else meta.get("chat_template"))

    if template is not None:
        template.bind(tokenizer)

    engine = InferenceModel(config, weights)
    del weights

    return TextGenerator(
        engine,
        tokenizer,
        default,
        max_batch_size=max_batch_size,
        cache_budget_bytes=cache_budget_bytes,
        prefill_chunk=prefill_chunk,
        chat_template=template,
        metadata={
            "weights_sha256": meta["weights"]["sha256"],
            "weights_dtype": meta["weights"]["dtype"],
            "source": meta.get("source", {}),
            "path": os.path.abspath(path),
            "context": describe_context(config),
        },
    )
