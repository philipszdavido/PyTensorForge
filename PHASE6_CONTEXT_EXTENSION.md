# Phase 6 — Context Extension for RoPE Models

A model trained with `position_encoding: rope` at context length N can be run
at a longer context L without retraining, using one of three methods. The
extension is applied when a model is exported or loaded. The weights never
change.

| Method | What it does |
|---|---|
| `extrapolate` | Uses the RoPE tables past N unchanged. Exact inside the trained length. |
| `linear` | Position interpolation: position p is treated as p·N/L, so every position maps into the trained range. |
| `ntk` | NTK-aware scaling: the RoPE base becomes θ·s^(d/(d−2)), with s = L/N and d the head dimension. Low frequencies stretch; the highest frequency is unchanged. |

Models with learned position tables (`position_encoding: learned`) cannot be
extended, because they have no embeddings past the trained length. Requests
to extend them are refused with an explanation.

## Usage

Bake the extension into an export:

```bash
pytensorforge export checkpoints/<ckpt> --output exports/mygpt-2k --tokenizer tokenizer.json \
    --extend-context 2048 --context-extension ntk
```

Or apply it when loading or serving an existing export:

```python
gen = load_model("exports/mygpt", context_length=2048, context_extension="ntk")
```

```yaml
models:
  - name: mygpt
    path: exports/mygpt
    extend_context_to: 2048
    context_extension: ntk
```

The model config records the method, factor and original trained length
(`rope_scaling`, `rope_scaling_factor`, `trained_context_length`). Extending
an already-extended model is computed relative to the original trained
length. `GET /v1/models` reports a `context` object with these details, and
all request limits (prompt budget, `max_tokens`, truncation) use the extended
length.

## How well does it work?

`python -m test.gpt.bench_context_extension` trains a small RoPE model
(d_model 96, 3 layers) at context 64 on ~137k tokens of English text. It then
measures held-out loss by position at length 256 (4× the trained length).
Two seeds:

| Method | Seed | 0–63 | 64–127 | 128–255 | All |
|---|---|---|---|---|---|
| extrapolate | 0 | 4.250 | 4.332 | 4.722 | 4.507 |
| linear | 0 | 4.547 | 4.581 | 4.631 | 4.598 |
| ntk | 0 | 4.269 | **4.252** | **4.547** | **4.404** |
| extrapolate | 1 | 4.316 | 4.612 | 4.828 | 4.646 |
| linear | 1 | 4.685 | 4.863 | 4.808 | 4.791 |
| ntk | 1 | 4.343 | **4.546** | **4.721** | **4.583** |

Held-out loss at the trained length was 4.24 (seed 0) and 4.36 (seed 1).

What the numbers support:

- **Use `ntk` if you extend without fine-tuning.** It had the lowest loss
  past the trained length in every bucket for both seeds, and it barely
  changes quality inside the trained length.
- **No method preserves trained-length quality at 4×.** The gain over plain
  extrapolation is real but modest. With seed 0, NTK looked nearly lossless up
  to 2×; with seed 1 it degraded noticeably there. Treat 2× as "usable with
  some loss", not "free".
- **`linear` without fine-tuning is the worst choice.** It hurts even short
  prompts, because it changes every position the model sees. Position
  interpolation is designed to be followed by a short fine-tune at the new
  length, which PyTensorForge does not yet support (see below).
- **Caveat on the evidence:** the model is small and overfits (train loss
  ~2.7 vs held-out ~4.3), and the held-out set is ~7k tokens. The ranking is
  consistent with the published results for these methods, but run the
  benchmark on your own model before relying on a long context.

The model can only use distant context as well as its training taught it to.
Extension lets it *accept* longer inputs; it doesn't teach long-range
reasoning.

## Correctness tests

`python -m test.gpt.test_phase6_context` checks:

- **Validation:** learned models, non-increasing lengths, unknown methods,
  factors below 1, and inconsistent settings are rejected. Re-extension
  measures from the original length. Configs written before Phase 6 load as
  unextended.
- **Table maths:** linear maps position 4p onto trained position p; the NTK
  base follows the formula and keeps the highest frequency; extrapolation is
  bit-identical inside the trained length.
- **Engine agreement:** for every method, the training model and the inference
  engine agree past the trained length, in chunked prefill and in batched
  decode crossing the trained boundary. An unextended engine still refuses
  positions past its context.
- **End to end:** export baking, load-time override, a long prompt generated
  past the trained length, truncation still applied without extension,
  serving-config validation, and the HTTP server reporting and serving the
  extended context.

All earlier suites (Phases 2–5, tokenizer, browser) pass.

## Not included

- **Fine-tuning at the extended length.** This is the standard way to make
  `linear` (and to improve `ntk`), but it needs a "start a new run from these
  weights" mode, which doesn't exist yet. Resuming a checkpoint with a changed
  architecture is refused by design.
- **Dynamic NTK and YaRN:** scaling that varies with the current sequence
  length, and per-frequency interpolation with attention temperature. Both are
  refinements of the methods here and are not implemented.
- **Memory growth:** KV-cache memory grows linearly with context. At 4× the
  length each request's cache is 4× larger, so revisit `kv_cache_budget_mb`
  and `max_batch_size` when serving an extended model.
