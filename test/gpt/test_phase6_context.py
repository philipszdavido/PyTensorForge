import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.Tensor import Tensor, no_grad
from src.inference.engine import InferenceModel
from src.inference.export import export_model, flatten_state
from src.inference.runtime import load_model
from src.models.gpt.config import GPTConfig
from src.models.gpt.context import describe_context, extend_context
from src.models.gpt.model import GPTModel
from src.models.gpt.rope import RotaryTables
from src.serving.config import server_config_from_dict
from test.gpt.test_phase4_server import auth, request, start_server
from test.gpt.test_phase5_training import build_trainer, check, make_corpus, make_tokenizer


def rope_config(ctx=32, vocab=60):
    return GPTConfig(vocab_size=vocab, context_length=ctx, d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                     position_encoding="rope", rope_theta=1000.0)


def perturbed_model(cfg, seed=0):
    np.random.seed(seed)
    model = GPTModel(cfg)
    model.build()
    rng = np.random.default_rng(seed)

    for p in model.parameters():
        p.data[:] = p.data + rng.normal(0, 0.3, p.data.shape).astype(np.float32)

    return model


def forward(model, ids):
    with no_grad():
        return model(Tensor(np.asarray(ids)[None], requires_grad=False)).data[0]


def test_config_and_tables():
    base = rope_config()

    for bad, why in (
        (lambda: extend_context(GPTConfig(vocab_size=10, context_length=32, d_model=16, n_heads=2), 64, "ntk"),
         "learned-position models cannot be extended"),
        (lambda: extend_context(base, 32, "ntk"), "extension must be longer than the trained context"),
        (lambda: extend_context(base, 64, "yarn"), "unknown methods are rejected"),
        (lambda: GPTConfig(vocab_size=10, context_length=32, d_model=16, n_heads=2, rope_scaling="ntk"),
         "scaling on a learned-position model is rejected"),
        (lambda: GPTConfig(vocab_size=10, context_length=32, d_model=16, n_heads=2, position_encoding="rope",
                           rope_scaling_factor=0.5), "factors below 1 are rejected"),
        (lambda: GPTConfig(vocab_size=10, context_length=32, d_model=16, n_heads=2, position_encoding="rope",
                           trained_context_length=64), "trained length cannot exceed the context"),
    ):
        try:
            bad()
            check(False, why)
        except ValueError:
            pass

    ext = extend_context(base, 128, "ntk")
    again = extend_context(ext, 256, "linear")
    check(again.trained_context_length == 32 and again.rope_scaling_factor == 8.0,
          "re-extending is relative to the original trained length")
    check(GPTConfig.from_dict(ext.to_dict()) == ext, "extended config round-trips")
    check(GPTConfig.normalized({k: v for k, v in base.to_dict().items()
                                if k not in ("rope_scaling", "rope_scaling_factor", "trained_context_length")})
          == base.to_dict(), "configs written before phase 6 normalize to no extension")
    check(describe_context(base) == {"context_length": 32, "extended": False}, "unextended description")
    check(describe_context(ext)["method"] == "ntk" and describe_context(ext)["factor"] == 4.0, "extended description")

    plain = RotaryTables(8, 32, 1000.0)
    linear = RotaryTables(8, 128, 1000.0, "linear", 4.0)
    ntk = RotaryTables(8, 128, 1000.0, "ntk", 4.0)
    extrapolate = RotaryTables(8, 128, 1000.0)
    check(np.allclose(linear.cos[::4][:32], plain.cos) and np.allclose(linear.sin[::4][:32], plain.sin),
          "linear scaling maps position 4p onto trained position p")
    check(abs(ntk.theta - 1000.0 * 4.0 ** (8 / 6)) < 1e-6, "ntk base follows theta * factor^(d/(d-2))")
    check(np.allclose(ntk.cos[:, 0], np.cos(np.arange(128)), atol=1e-6),
          "ntk keeps the highest frequency unchanged")
    check(np.array_equal(extrapolate.cos[:32], plain.cos), "extrapolation keeps trained positions unchanged")
    print("context extension: config validation, re-extension, round-trip, table maths")


def test_models_agree():
    base = rope_config()
    model = perturbed_model(base)
    state = model.state_dict()
    rng = np.random.default_rng(3)
    short = rng.integers(0, 60, 24)
    long = rng.integers(0, 60, 100)

    for method in ("extrapolate", "linear", "ntk"):
        cfg = extend_context(base, 128, method)
        ext = GPTModel(cfg)
        ext.build()
        ext.load_state_dict(state)

        if method == "extrapolate":
            check(np.array_equal(forward(ext, short), forward(model, short)),
                  "extrapolation is exact inside the trained length")

        full = forward(ext, long)
        check(np.isfinite(full).all(), f"{method}: finite logits past the trained length")

        engine = InferenceModel(cfg, flatten_state(state))
        cache = engine.new_cache(128)
        last = engine.prefill(long, cache, chunk_size=13)
        check(np.allclose(last, full[-1], atol=2e-4), f"{method}: engine prefill matches training model at 100 tokens")

        caches = [engine.new_cache(128), engine.new_cache(128)]
        engine.prefill(long[:40], caches[0])
        engine.prefill(long[:70], caches[1])
        worst = 0.0

        for step in range(25):
            logits = engine.decode_batch([long[40 + step], long[70 + step]], caches)
            worst = max(worst, np.abs(logits[0] - full[40 + step]).max(), np.abs(logits[1] - full[70 + step]).max())

        check(worst < 2e-4, f"{method}: batched decode across the trained boundary matches ({worst:.1e})")

    plain_engine = InferenceModel(base, flatten_state(state))
    capped = plain_engine.new_cache(128)
    check(capped.capacity == 32, "unextended engine caps caches at its trained context")

    try:
        plain_engine.prefill(long, capped)
        check(False, "unextended engine refuses positions past its context")
    except ValueError:
        pass

    print("context extension: training model and engine agree past the trained length for every method")


def test_export_load_serve(tmp):
    corpus = os.path.join(tmp, "corpus")
    make_corpus(corpus)
    tok, tok_path = make_tokenizer(tmp, corpus)
    trainer, _ = build_trainer(tok, corpus, os.path.join(tmp, "ckpt"), 3, model={"position_encoding": "rope"})
    trainer.train()
    ckpt = trainer.checkpoints.latest_path()

    plain_dir = os.path.join(tmp, "plain")
    export_model(ckpt, plain_dir, tok_path)
    baked_dir = os.path.join(tmp, "baked")
    export_model(ckpt, baked_dir, tok_path, context_length=64, context_extension="ntk")

    with open(os.path.join(baked_dir, "config.json")) as f:
        meta = json.load(f)["model"]

    check(meta["context_length"] == 64 and meta["trained_context_length"] == 16 and meta["rope_scaling"] == "ntk",
          "export bakes the extension into the model config")

    long_prompt = " ".join(["the quick brown fox jumps over the lazy dog"] * 4)

    for path, overrides in ((baked_dir, {}), (plain_dir, {"context_length": 64, "context_extension": "ntk"})):
        gen = load_model(path, max_batch_size=2, **overrides)

        try:
            check(gen.config.context_length == 64, "loaded model has the extended context")
            check(gen.metadata["context"]["extended"] and gen.metadata["context"]["trained_context_length"] == 16,
                  "runtime metadata reports the extension")
            prompt_len = len(tok.encode(long_prompt))
            check(prompt_len > 16, "prompt is longer than the trained context")
            result = gen.generate(long_prompt, max_new_tokens=8, temperature=0.0)
            check(result.error is None and result.usage["prompt_tokens"] == prompt_len,
                  "generation with a prompt past the trained length")
        finally:
            gen.stop()

    plain = load_model(plain_dir, max_batch_size=2)

    try:
        result = plain.generate(long_prompt, max_new_tokens=8, temperature=0.0, truncate_prompt=True)
        check(result.usage["prompt_tokens"] < 16, "unextended model still truncates to its trained context")
    finally:
        plain.stop()

    try:
        load_model(plain_dir, context_length=64)
        check(False, "context_length without a method is rejected")
    except ValueError:
        pass

    for bad in ({"extend_context_to": 64}, {"context_extension": "ntk"}, {"extend_context_to": 64, "context_extension": "x"}):
        try:
            server_config_from_dict({"models": [{"name": "m", "path": plain_dir, **bad}]}).validate()
            check(False, f"invalid serving extension settings accepted: {bad}")
        except ValueError:
            pass

    srv = start_server(plain_dir, models=[{"name": "tiny", "path": plain_dir, "max_batch_size": 2,
                                           "extend_context_to": 64, "context_extension": "ntk"}])

    try:
        resp, body = request(srv.port, "GET", "/v1/models/tiny", headers=auth())
        info = json.loads(body)
        check(resp.status == 200 and info["context_length"] == 64 and info["context"]["method"] == "ntk",
              "server reports the extended context")
        resp, body = request(srv.port, "POST", "/v1/completions", headers=auth(),
                             body={"model": "tiny", "prompt": long_prompt, "max_tokens": 8, "temperature": 0})
        usage = json.loads(body)["usage"]
        check(resp.status == 200 and usage["prompt_tokens"] > 16, "server completes a prompt past the trained length")
    finally:
        srv.stop()

    print("context extension: export baking, load-time override, truncation without extension, serving end to end")


def main():
    tmp = tempfile.mkdtemp()

    try:
        test_config_and_tables()
        test_models_agree()
        test_export_load_serve(tmp)
        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
