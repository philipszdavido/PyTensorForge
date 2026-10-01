import gc
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import CheckpointSpec, DataSpec, EvaluationSpec, ModelSpec, RuntimeSpec, TrainConfig, TrainingSpec
from src.core.Tensor import Tensor, no_grad
from src.data.corpus import CorpusIndex
from src.data.streaming_dataset import StreamingTextDataset
from src.inference.engine import InferenceModel
from src.inference.export import export_model, flatten_state
from src.inference.runtime import load_model
from src.loss.CrossEntropyWithLogitsLoss import CrossEntropyWithLogitsLoss
from src.models.gpt.attention import CausalSelfAttention, causal_attention
from src.models.gpt.config import GPTConfig
from src.models.gpt.model import GPTModel
from src.models.gpt.rope import rotary_tables
from src.models.transformers.MultiHeadAttention import MultiHeadAttention
from src.tokenization.bpe import PTFBPETokenizer
from src.tokenization.bytebpe import train_byte_bpe
from src.training.precision import GradScaler, autocast, round_bf16, round_fp16
from src.training.trainer import Trainer

WORDS = "the quick brown fox jumps over lazy dog cat mat mouse house python code model token stream shard".split()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_corpus(root, seed=0, files=4, lines=300):
    rng = random.Random(seed)
    os.makedirs(root)

    for i in range(files):
        with open(os.path.join(root, f"f{i}.txt"), "w") as f:
            for _ in range(lines):
                f.write(" ".join(rng.choice(WORDS) for _ in range(12)) + "\n")

    with open(os.path.join(root, "g0.jsonl"), "w") as f:
        for _ in range(120):
            f.write(json.dumps({"text": "Ünïcødé " + " ".join(rng.choice(WORDS) for _ in range(20))}) + "\n")


def make_tokenizer(tmp, corpus_dir):
    texts = []

    for p in CorpusIndex(corpus_dir).files:
        with open(p) as f:
            texts.append(f.read()[:5000])

    tok = PTFBPETokenizer(vocab_size=120)
    tok.fit(texts)
    path = os.path.join(tmp, "tok.json")
    tok.save(path)
    return PTFBPETokenizer.load(path), path


def directional_check(fn, inputs, seed=0, eps=1e-2):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal(fn(*inputs).shape).astype(np.float32)

    for t in inputs:
        t.grad = None

    out = fn(*inputs)
    (out * Tensor(weight, requires_grad=False)).sum().backward()
    grads = [t.grad.copy() for t in inputs]

    worst = 0.0

    for i, t in enumerate(inputs):
        direction = rng.standard_normal(t.shape).astype(np.float32)
        base = t.data.copy()

        t.data[:] = base + eps * direction
        with no_grad():
            plus = float(np.sum(fn(*inputs).data.astype(np.float64) * weight))

        t.data[:] = base - eps * direction
        with no_grad():
            minus = float(np.sum(fn(*inputs).data.astype(np.float64) * weight))

        t.data[:] = base
        numeric = (plus - minus) / (2 * eps)
        analytic = float(np.sum(grads[i].astype(np.float64) * direction))
        worst = max(worst, abs(numeric - analytic) / max(1e-3, abs(numeric), abs(analytic)))

    return worst


def test_tensor_core():
    a = Tensor(np.ones((2, 3)), requires_grad=True)
    check(a._grad is None, "gradient buffers are allocated lazily")

    b = (a * 2.0).sum()
    b.backward()
    check(np.allclose(a.grad, 2.0), "gradient through lazy buffers")

    with no_grad():
        c = a * 3.0 + 1.0
    check(not c.requires_grad and c.parents == (), "no_grad builds no graph")

    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    w = Tensor(np.random.randn(5, 3), requires_grad=True)
    h = x @ w
    y = h.gelu().sum()
    y.backward(release=True)
    check(h.parents == () and h._grad is None, "release frees intermediate graph and gradients")
    check(w._grad is not None and x._grad is not None, "release keeps leaf gradients")

    ref = np.random.randn(1000).astype(np.float32) * 4
    out = Tensor(ref).gelu()
    exact = 0.5 * ref.astype(np.float64) * (1 + np.tanh(np.sqrt(2 / np.pi) * (ref + 0.044715 * ref.astype(np.float64) ** 3)))
    check(out.data.dtype == np.float32 and np.abs(out.data - exact).max() < 1e-5, "gelu value in float32")
    err = directional_check(lambda t: t.gelu(), [Tensor(np.random.randn(6, 7), requires_grad=True)])
    check(err < 1e-2, f"gelu gradient (rel err {err:.2e})")

    deep = Tensor(np.ones(3), requires_grad=True)
    z = deep

    for _ in range(3000):
        z = z + 1.0

    z.sum().backward()
    check(np.allclose(deep.grad, 1.0), "iterative topological sort handles deep graphs")
    print("tensor core: lazy grads, no_grad, graph release, gelu float32, deep graphs ok")


def test_attention_equivalence():
    rng = np.random.default_rng(1)
    B, T, d, H = 2, 7, 16, 4
    x_data = rng.standard_normal((B, T, d)).astype(np.float32)

    legacy = MultiHeadAttention(d, H)
    legacy.build((B, T, d))
    fused = CausalSelfAttention(d, H)
    fused.build((B, T, d))
    fused.load_state_dict(legacy.state_dict())

    mask = Tensor(np.tril(np.ones((T, T), dtype=np.float32)).reshape(1, 1, T, T), requires_grad=False)
    weight = rng.standard_normal((B, T, d)).astype(np.float32)

    results = []

    for layer, kwargs in ((legacy, {"mask": mask}), (fused, {})):
        x = Tensor(x_data.copy(), requires_grad=True)
        out = layer.call(x, **kwargs)
        (out * Tensor(weight, requires_grad=False)).sum().backward()
        results.append((out.data.copy(), x.grad.copy(), [p.grad.copy() for p in (layer.Wq, layer.Wk, layer.Wv, layer.Wo)]))

        for p in (layer.Wq, layer.Wk, layer.Wv, layer.Wo):
            p.grad = None

    (o1, gx1, gw1), (o2, gx2, gw2) = results
    check(np.allclose(o1, o2, atol=1e-5), "fused attention forward equals legacy attention")
    check(np.allclose(gx1, gx2, atol=1e-4), "fused attention input gradient equals legacy")
    check(all(np.allclose(a, b, atol=1e-4) for a, b in zip(gw1, gw2)), "fused attention weight gradients equal legacy")

    rope = rotary_tables(d // H, 32)

    for use_rope in (None, rope):
        q = Tensor(rng.standard_normal((B, T, d)), requires_grad=True)
        k = Tensor(rng.standard_normal((B, T, d)), requires_grad=True)
        v = Tensor(rng.standard_normal((B, T, d)), requires_grad=True)
        err = directional_check(lambda q, k, v: causal_attention(q, k, v, H, use_rope), [q, k, v])
        check(err < 2e-2, f"causal attention gradient (rope={use_rope is not None}, rel err {err:.2e})")

    q = Tensor(rng.standard_normal((1, T, d)), requires_grad=False)
    k = Tensor(rng.standard_normal((1, T, d)), requires_grad=False)
    v = Tensor(rng.standard_normal((1, T, d)), requires_grad=False)
    base = causal_attention(q, k, v, H, rope).data
    q.data[:, 3:] += 100.0
    k.data[:, 5:] += 100.0
    v.data[:, 5:] += 100.0
    later = causal_attention(q, k, v, H, rope).data
    check(np.allclose(base[:, :3], later[:, :3], atol=1e-5), "attention is causal")
    print("attention: fused == legacy (values and gradients), gradient checks with and without rope, causal")


def test_rope_properties():
    tables = rotary_tables(8, 64, 10000.0)
    rng = np.random.default_rng(3)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)

    def score(i, j):
        ci, si = tables.tables([i])
        cj, sj = tables.tables([j])
        return float(tables.rotate(a[None], ci, si)[0] @ tables.rotate(b[None], cj, sj)[0])

    check(abs(score(3, 1) - score(40, 38)) < 1e-4 and abs(score(10, 10) - score(0, 0)) < 1e-4,
          "rope scores depend only on relative position")

    c, s = tables.tables(np.arange(10))
    x = rng.standard_normal((10, 8)).astype(np.float32)
    check(np.allclose(tables.rotate(tables.rotate(x, c, s), c, s, inverse=True), x, atol=1e-6), "rope inverse")
    check(np.allclose(np.linalg.norm(tables.rotate(x, c, s), axis=-1), np.linalg.norm(x, axis=-1), atol=1e-5),
          "rope preserves norms")

    try:
        tables.tables([64])
        check(False, "positions past the table are rejected")
    except ValueError:
        pass

    for bad in ({"d_model": 12, "n_heads": 4, "position_encoding": "rope"}, {"position_encoding": "alibi"}):
        try:
            GPTConfig(vocab_size=10, **{"d_model": 16, "n_heads": 4, **bad})
            check(False, f"invalid config accepted: {bad}")
        except ValueError:
            pass

    print("rope: relative-position invariance, inverse, norm preservation, config validation")


def rope_model(vocab, seed=0, ctx=40, tie=True):
    cfg = GPTConfig(vocab_size=vocab, context_length=ctx, d_model=32, n_layers=2, n_heads=4, ff_dim=64,
                    tie_weights=tie, position_encoding="rope", rope_theta=500.0)
    np.random.seed(seed)
    model = GPTModel(cfg)
    model.build()
    rng = np.random.default_rng(seed)

    for p in model.parameters():
        p.data[:] = p.data + rng.normal(0, 0.3, p.data.shape).astype(np.float32)

    return cfg, model


def test_rope_inference_matches_training(vocab):
    for tie in (True, False):
        cfg, model = rope_model(vocab, tie=tie)
        check("pos_emb" not in model.state_dict(), "rope model has no learned position table")
        engine = InferenceModel(cfg, flatten_state(model.state_dict()))

        rng = np.random.default_rng(5)
        seq = rng.integers(0, vocab, 30)

        with no_grad():
            full = model(Tensor(seq[None], requires_grad=False)).data[0]

        cache = engine.new_cache(40)
        last = engine.prefill(seq, cache, chunk_size=7)
        check(np.allclose(last, full[-1], atol=2e-4), "rope prefill logits equal training forward")

        caches = [engine.new_cache(40), engine.new_cache(40)]
        engine.prefill(seq[:10], caches[0])
        engine.prefill(seq[:4], caches[1])
        worst = 0.0

        for step in range(12):
            logits = engine.decode_batch([seq[10 + step], seq[4 + step]], caches)
            worst = max(worst, np.abs(logits[0] - full[10 + step]).max(), np.abs(logits[1] - full[4 + step]).max())

        check(worst < 2e-4, f"rope batched decode at mixed positions equals training forward ({worst:.2e})")

    print("rope inference: chunked prefill and mixed-position batched decode match the training model (tied/untied)")


def test_engine_stays_float32():
    rng = np.random.default_rng(7)

    for act in ("gelu", "relu", "tanh", "sigmoid"):
        for encoding in ("learned", "rope"):
            cfg = GPTConfig(vocab_size=50, context_length=32, d_model=16, n_layers=2, n_heads=2, ff_dim=32,
                            activation=act, position_encoding=encoding)
            np.random.seed(0)
            model = GPTModel(cfg)
            model.build()
            engine = InferenceModel(cfg, flatten_state(model.state_dict()))
            cache = engine.new_cache(32)
            prefill = engine.prefill(rng.integers(0, 50, 5), cache)
            decode = engine.decode_batch([3], [cache])
            check(prefill.dtype == np.float32 and decode.dtype == np.float32,
                  f"inference engine stays float32 ({act}, {encoding}): {prefill.dtype}, {decode.dtype}")

    print("inference engine: float32 end to end for every activation and position encoding")


def make_config(tok, ckpt, total_steps, **training):
    model = training.pop("model", {})
    data = training.pop("data", {})
    return TrainConfig(
        model=ModelSpec(vocab_size=tok.vocab_size, context_length=16, d_model=16, n_layers=2, n_heads=2, ff_dim=32, **model),
        data=DataSpec(sequence_length=16, prefetch=0, **data),
        training=TrainingSpec(total_steps=total_steps, micro_batch_size=2, gradient_accumulation_steps=2,
                              learning_rate=3e-3, warmup_steps=2, decay="cosine", max_epochs=5, **training),
        checkpoint=CheckpointSpec(directory=ckpt, interval_steps=1000, keep_last=2),
        evaluation=EvaluationSpec(interval_steps=0),
        runtime=RuntimeSpec(),
    )


def build_trainer(tok, corpus_dir, ckpt, total_steps, workers=1, **training):
    cfg = make_config(tok, ckpt, total_steps, **training)
    m = cfg.model
    gcfg = GPTConfig(vocab_size=tok.vocab_size, context_length=m.context_length, d_model=m.d_model, n_layers=m.n_layers,
                     n_heads=m.n_heads, ff_dim=m.ff_dim, dropout=m.dropout,
                     position_encoding=m.position_encoding, rope_theta=m.rope_theta)
    random.seed(1)
    np.random.seed(1)
    model = GPTModel(gcfg)
    model.build()
    ds = StreamingTextDataset(CorpusIndex(corpus_dir), tok, 16, read_buffer_size=257, workers=workers)
    return Trainer(model, ds, cfg, tok, install_signal_handlers=False), model


def flat(model):
    return np.concatenate([p.data.ravel() for p in model.parameters()])


def losses(ckpt):
    out = []

    with open(os.path.join(ckpt, "train_log.jsonl")) as f:
        for line in f:
            rec = json.loads(line)

            if "loss" in rec and "skipped" not in rec:
                out.append(rec["loss"])

    return out


def resume_matches(tmp, tok, corpus_dir, name, total=6, split=3, **training):
    ref_dir = os.path.join(tmp, f"{name}_ref")
    t_ref, m_ref = build_trainer(tok, corpus_dir, ref_dir, total, **training)
    state0 = m_ref.state_dict()
    t_ref.train()

    run_dir = os.path.join(tmp, f"{name}_run")
    t1, m1 = build_trainer(tok, corpus_dir, run_dir, total, **training)
    m1.load_state_dict(state0)
    log_step = t1._log

    def interrupt_after_split(*args):
        log_step(*args)

        if t1.optimizer_steps >= split:
            t1.request_stop()

    t1._log = interrupt_after_split
    t1.train()
    check(t1.optimizer_steps == split, f"{name}: run interrupted mid-way")

    t2, m2 = build_trainer(tok, corpus_dir, run_dir, total, **training)
    check(t2.load_checkpoint(), "checkpoint found")
    t2.train()

    check(t2.optimizer_steps == t_ref.optimizer_steps and t2.tokens_processed == t_ref.tokens_processed,
          f"{name}: resumed run reaches the same step and token count")
    check(np.array_equal(flat(m2), flat(m_ref)), f"{name}: resumed weights are bit-identical to an uninterrupted run")
    return t_ref, m_ref, ref_dir


def test_training_features(tmp, tok, corpus_dir):
    t, m, d = resume_matches(tmp, tok, corpus_dir, "rope", model={"position_encoding": "rope"})
    ls = losses(d)
    check(ls[-1] < ls[0], f"rope model learns ({ls[0]:.3f} -> {ls[-1]:.3f})")
    print(f"rope training: loss {ls[0]:.3f} -> {ls[-1]:.3f}; resume is bit-identical")

    export_dir = os.path.join(tmp, "rope_export")
    export_model(t.checkpoints.latest_path(), export_dir, os.path.join(tmp, "tok.json"))
    gen = load_model(export_dir, max_batch_size=2)

    try:
        prompt = tok.encode("the quick brown")
        with no_grad():
            m.set_training(False)
            ref = m(Tensor(np.array([prompt]), requires_grad=False)).data[0, -1]
        cache = gen.engine.new_cache(16)
        got = gen.engine.prefill(prompt, cache)
        check(np.allclose(ref, got, atol=2e-4), "exported rope model reproduces the trained model")
        result = gen.generate("the quick", max_new_tokens=5, temperature=0.0)
        check(result.error is None and result.usage["completion_tokens"] > 0, "exported rope model generates")
    finally:
        gen.stop()

    print("rope export: checkpoint -> export -> load_model reproduces training logits and generates")

    _, _, d_ckpt = resume_matches(tmp, tok, corpus_dir, "actckpt", activation_checkpointing=True,
                                  model={"dropout": 0.1})
    _, _, d_plain = resume_matches(tmp, tok, corpus_dir, "plain", model={"dropout": 0.1})
    check(np.allclose(losses(d_ckpt), losses(d_plain), rtol=0, atol=1e-6),
          "activation checkpointing (with dropout) trains identically to the plain run")
    print("activation checkpointing: identical loss curve with dropout; resume bit-identical")

    for precision in ("bf16", "fp16"):
        t, _, d = resume_matches(tmp, tok, corpus_dir, precision, total=8, split=4, precision=precision)
        ls = losses(d)
        check(all(np.isfinite(ls)) and ls[-1] < ls[0], f"{precision} training is stable and learns")
        check(t.scaler.enabled == (precision == "fp16"), f"{precision} loss scaling resolved automatically")
        print(f"{precision}: loss {ls[0]:.3f} -> {ls[-1]:.3f}; resume bit-identical (loss scaler state restored)")

    fp32_dir = os.path.join(tmp, "fp32cmp")
    t32, m32 = build_trainer(tok, corpus_dir, fp32_dir, 4)
    t32.train()
    t16, m16 = build_trainer(tok, corpus_dir, os.path.join(tmp, "bf16cmp"), 4, precision="bf16")
    t16.train()
    diff = np.abs(flat(m32) - flat(m16)).max()
    check(0 < diff < 5e-2, f"bf16 changes numerics, but only slightly ({diff:.2e})")

    over_dir = os.path.join(tmp, "overflow")
    t, m = build_trainer(tok, corpus_dir, over_dir, 6, precision="fp16", initial_loss_scale=2.0 ** 24,
                         loss_scale_growth_interval=3)
    before = flat(m).copy()
    t.train()
    check(t.scaler.skipped_steps > 0, "fp16 overflow detected")
    check(t.optimizer_steps == 6 and np.isfinite(flat(m)).all() and not np.array_equal(before, flat(m)),
          "training recovers after overflow and weights stay finite")

    with open(os.path.join(over_dir, "train_log.jsonl")) as f:
        recs = [json.loads(l) for l in f]

    skipped = [r for r in recs if r.get("skipped")]
    check(len(skipped) == t.scaler.skipped_steps and skipped[0]["loss_scale"] < 2.0 ** 24, "skipped steps are logged")
    print(f"fp16 overflow: {t.scaler.skipped_steps} steps skipped, scale backed off to {t.scaler.scale:g}, training recovered")


def test_precision_units():
    x = np.array([1.0, 1 + 2 ** -8, 1 + 3 * 2 ** -8, -2.5, np.inf, -np.inf, 3.4e38, 0.0], np.float32)
    got = round_bf16(x)
    want = np.array([1.0, 1.0, 1 + 2 ** -6, -2.5, np.inf, -np.inf, np.inf, 0.0], np.float32)
    check(np.array_equal(got, want), f"bf16 round-to-nearest-even ({got})")
    check(np.isnan(round_bf16(np.array([np.nan], np.float32)))[0], "bf16 keeps NaN")
    check((round_bf16(np.random.randn(1000).astype(np.float32)).view(np.uint32) & 0xFFFF == 0).all(),
          "bf16 values have 8-bit mantissas")
    check(round_fp16(np.array([70000.0], np.float32))[0] == np.inf, "fp16 overflow becomes inf")

    a = Tensor(np.random.randn(8, 8), requires_grad=True)
    b = Tensor(np.random.randn(8, 8), requires_grad=True)

    with autocast("bf16"):
        c = a @ b

    check((c.data.view(np.uint32) & 0xFFFF == 0).all(), "autocast rounds matmul outputs")
    check(not np.array_equal(c.data, a.data @ b.data), "autocast changes the numerics")
    d = a @ b
    check(np.allclose(d.data, a.data @ b.data), "autocast is scoped")

    s = GradScaler(init_scale=1024.0, growth_interval=2)
    p = Tensor(np.ones(3), requires_grad=True)
    p.grad = np.array([1.0, np.inf, 0.0], np.float32)
    ok = s.unscale_and_check([p])
    s.update(not ok)
    check(not ok and s.scale == 512.0 and s.skipped_steps == 1, "scaler backs off on overflow")
    p.grad = np.array([512.0, 1024.0, 0.0], np.float32)
    ok = s.unscale_and_check([p])
    s.update(not ok)
    check(ok and np.allclose(p.grad, [1, 2, 0]), "scaler unscales gradients")
    s.update(False)
    check(s.scale == 1024.0, "scaler grows after growth_interval clean steps")
    s2 = GradScaler(init_scale=1.0)
    s2.load_state_dict(s.state_dict())
    check(s2.state_dict() == s.state_dict(), "scaler state round-trips")

    try:
        GradScaler(enabled=False).load_state_dict(s.state_dict())
        check(False, "scaler mode mismatch rejected")
    except ValueError:
        pass

    print("precision: bf16/fp16 rounding, scoped autocast, loss scaler overflow/backoff/growth/state")


def grads_for(model, x, y):
    for p in model.parameters():
        p.grad = None

    logits = model(Tensor(x, requires_grad=False))
    b, s, v = logits.shape
    loss = CrossEntropyWithLogitsLoss()(logits.reshape(b * s, v), Tensor(y, requires_grad=False).reshape(b * s))
    loss.backward(release=True)
    return float(loss.data), [p.grad.copy() for p in model.parameters()]


def test_activation_checkpointing_memory():
    cfg = GPTConfig(vocab_size=256, context_length=96, d_model=96, n_layers=6, n_heads=4, ff_dim=384, dropout=0.1)
    np.random.seed(0)
    model = GPTModel(cfg)
    model.build()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (4, 96))
    y = rng.integers(0, 256, (4, 96))

    results = {}

    for flag in (False, True):
        model.activation_checkpointing = flag
        np.random.seed(42)
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        loss, grads = grads_for(model, x, y)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[flag] = (loss, grads, peak, elapsed, np.random.get_state()[1].copy())

    (l0, g0, p0, e0, r0), (l1, g1, p1, e1, r1) = results[False], results[True]
    check(l0 == l1, "checkpointed forward gives the same loss")
    check(all(np.array_equal(a, b) for a, b in zip(g0, g1)), "checkpointed gradients are bit-identical (dropout replayed)")
    check(np.array_equal(r0, r1), "recomputation does not disturb the global RNG stream")
    check(p1 < 0.6 * p0, f"activation checkpointing lowers peak memory ({p0 / 2**20:.1f} -> {p1 / 2**20:.1f} MiB)")
    print(f"activation checkpointing: peak {p0 / 2**20:.1f} -> {p1 / 2**20:.1f} MiB "
          f"({p1 / p0:.0%}), step {e0:.2f}s -> {e1:.2f}s, gradients bit-identical")


def child_processes():
    return len(mp.active_children())


def test_parallel_tokenization(tmp, corpus_dir):
    tok_dir = os.path.join(tmp, "bbpe")
    texts = []

    for p in CorpusIndex(corpus_dir).files:
        with open(p, encoding="utf-8") as f:
            texts.append(f.read())

    bbpe = train_byte_bpe(texts, vocab_size=400)
    bbpe_path = os.path.join(tmp, "bbpe.json")
    bbpe.save(bbpe_path)

    from src.tokenization.registry import load_tokenizer
    bbpe = load_tokenizer(bbpe_path)
    os.makedirs(tok_dir)
    word_tok, _ = make_tokenizer(tok_dir, corpus_dir)

    for tok in (word_tok, bbpe):
        def run(workers, limit=None, state=None):
            ds = StreamingTextDataset(CorpusIndex(corpus_dir), tok, 16, read_buffer_size=257, workers=workers)

            if state is not None:
                ds.load_state_dict(state)

            out = []
            gen = ds.batches(4)

            for x, _ in gen:
                out.append(x.copy())

                if limit and len(out) >= limit:
                    break

            gen.close()
            return out, ds.state_dict()

        serial, _ = run(1)
        start = time.perf_counter()
        parallel, _ = run(3)
        par_s = time.perf_counter() - start
        check(len(serial) == len(parallel) and all(np.array_equal(a, b) for a, b in zip(serial, parallel)),
              f"parallel tokenization yields identical batches ({type(tok).__name__})")

        head, state = run(3, limit=7)
        tail, _ = run(3, state=json.loads(json.dumps(state)))
        joined = head + tail
        check(len(joined) == len(serial) and all(np.array_equal(a, b) for a, b in zip(serial, joined)),
              "parallel tokenization resumes exactly from a mid-stream state")
        _, serial_state = run(1, limit=7)
        check(serial_state == state, "dataset state is identical for serial and parallel encoding")

    deadline = time.time() + 5

    while child_processes() and time.time() < deadline:
        time.sleep(0.05)

    check(child_processes() == 0, "tokenizer worker processes are cleaned up")
    print(f"parallel mode A: identical batches and resume state for word/byte BPE; last run {par_s:.2f}s; workers reaped")


def test_old_checkpoint_compat(tmp, tok, corpus_dir):
    d = os.path.join(tmp, "legacy")
    t, m = build_trainer(tok, corpus_dir, d, 2)
    t.train()
    path = t.checkpoints.latest_path()
    payload = t.checkpoints.load(path)

    for key in ("position_encoding", "rope_theta"):
        payload["model_config"].pop(key)

    payload.pop("grad_scaler_state")
    t.checkpoints.save(3, payload)

    t2, _ = build_trainer(tok, corpus_dir, d, 4)
    check(t2.load_checkpoint(), "checkpoint written before phase 5 (no rope/scaler fields) still loads")
    t2.train()
    check(t2.optimizer_steps == 4, "and training continues")

    t3, _ = build_trainer(tok, corpus_dir, d, 4, model={"position_encoding": "rope"})

    try:
        t3.load_checkpoint()
        check(False, "learned-position checkpoint must not load into a rope model")
    except ValueError:
        pass

    print("compatibility: pre-phase-5 checkpoints resume; architecture mismatch is refused")


def main():
    tmp = tempfile.mkdtemp()

    try:
        corpus_dir = os.path.join(tmp, "corpus")
        make_corpus(corpus_dir)
        tok, _ = make_tokenizer(tmp, corpus_dir)

        test_tensor_core()
        test_attention_equivalence()
        test_rope_properties()
        test_rope_inference_matches_training(tok.vocab_size)
        test_engine_stays_float32()
        test_precision_units()
        test_activation_checkpointing_memory()
        test_training_features(tmp, tok, corpus_dir)
        test_old_checkpoint_compat(tmp, tok, corpus_dir)
        test_parallel_tokenization(tmp, corpus_dir)
        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
