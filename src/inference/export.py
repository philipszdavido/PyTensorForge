import hashlib
import json
import os
import shutil

import numpy as np

from src.inference.config import GenerationConfig
from src.models.gpt.config import GPTConfig
from src.models.gpt.context import extend_context
from src.tokenization.base import identity_matches
from src.training.checkpoint_manager import CheckpointManager

EXPORT_FORMAT_VERSION = 1
WEIGHTS_FILE = os.path.join("weights", "model.npz")


def flatten_state(state, prefix=""):
    out = {}

    for key, value in state.items():
        if isinstance(value, dict):
            out.update(flatten_state(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                out.update(flatten_state(item, f"{prefix}{key}.{i}."))
        else:
            out[f"{prefix}{key}"] = np.asarray(value)

    return out


def _sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def _write_json(path, payload):
    tmp = path + ".tmp"

    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    os.replace(tmp, path)


def export_model(checkpoint_path, output_dir, tokenizer_path, generation=None, dtype="float32", chat_template=None,
                 context_length=None, context_extension=None):
    if dtype not in ("float32", "float16"):
        raise ValueError("dtype must be float32 or float16")

    payload = CheckpointManager(os.path.dirname(os.path.abspath(checkpoint_path))).load(checkpoint_path)

    if payload is None:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    from src.tokenization.registry import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)

    if not identity_matches(payload["tokenizer_identity"], tokenizer.identity):
        raise ValueError("tokenizer does not match the one this checkpoint was trained with")

    config = GPTConfig.from_dict(payload["model_config"])

    if context_length is not None or context_extension is not None:
        if context_length is None or context_extension is None:
            raise ValueError("context extension needs both context_length and context_extension")

        config = extend_context(config, context_length, context_extension)

    from src.inference.chat_template import ChatTemplate

    data_config = payload.get("data_config") or {}

    if chat_template is None and data_config.get("format") == "chat":
        chat_template = data_config.get("chat_template")

    template = ChatTemplate.resolve(chat_template)

    if template is not None:
        template.bind(tokenizer)
    weights = {k: v.astype(dtype) for k, v in flatten_state(payload["model_state"]).items()}

    os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)

    weights_path = os.path.join(output_dir, WEIGHTS_FILE)
    tmp_weights = weights_path + ".tmp"

    with open(tmp_weights, "wb") as f:
        np.savez(f, **weights)

    os.replace(tmp_weights, weights_path)

    shutil.copyfile(tokenizer_path, os.path.join(output_dir, "tokenizer", "tokenizer.json"))

    generation = generation or GenerationConfig(
        max_new_tokens=128, temperature=0.8, top_k=40, top_p=0.95,
    )

    _write_json(os.path.join(output_dir, "generation.json"), generation.to_dict())

    meta = {
        "format_version": EXPORT_FORMAT_VERSION,
        "architecture": config.architecture,
        "model": config.to_dict(),
        "tokenizer": {"file": "tokenizer/tokenizer.json", "identity": tokenizer.identity},
        "weights": {
            "file": WEIGHTS_FILE,
            "dtype": dtype,
            "sha256": _sha256(weights_path),
            "tensors": {k: list(v.shape) for k, v in weights.items()},
        },
        "source": {
            "framework_version": payload.get("framework_version"),
            "run_id": payload.get("run_id"),
            "optimizer_steps": payload.get("optimizer_steps"),
            "tokens_processed": payload.get("tokens_processed"),
        },
    }

    if template is not None:
        meta["chat_template"] = template.to_dict()

    _write_json(os.path.join(output_dir, "config.json"), meta)

    return output_dir
