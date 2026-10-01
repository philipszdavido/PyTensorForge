# PyTensorForge — Phase 3: Inference Runtime

## Pipeline

`train checkpoint (pickle)` → `export` → `model/` (no optimizer state, no pickle) → `load_model()` → `TextGenerator`

```
model/
  config.json        architecture, tokenizer identity, weight tensor shapes, SHA-256
  weights/model.npz  float32 or float16, loaded with allow_pickle=False
  tokenizer/tokenizer.json
  generation.json    default sampling settings
```

```python
from src.inference.runtime import load_model
model = load_model("model/")
result = model.generate("the quick", max_new_tokens=32, temperature=0.8, top_k=40)
for piece in model.generate_stream("Hello"):
    print(piece, end="")
```

CLI: `export CHECKPOINT --output DIR --tokenizer T [--dtype float16]`,
`generate DIR --prompt ... [--greedy --seed --temperature --top-k --top-p --repetition-penalty --stop S --timeout S]`.

## Components (`src/inference/`)

- `engine.py` — `InferenceModel`: NumPy-only forward with no autograd graph. Q/K/V are fused into one matmul.
  `prefill` (chunked, bounds attention memory), `forward_chunk`, and `decode_batch` (one new token for each of
  B sequences with *different* lengths; projections and FFN are batched, attention runs per sequence against its own cache).
  Only the last position's logits are ever computed.
- `kv_cache.py` — `KVCache` preallocates `(layers, heads, capacity, head_dim)` once per request and is never
  reallocated as tokens are appended. `KVCacheManager` enforces a byte budget (admission control).
- `scheduler.py` — `BatchScheduler`: continuous batching. Requests are admitted as cache budget and batch slots allow,
  prefilled, then decoded together each step; finished sequences leave immediately and free their cache.
  `RequestHandle` gives per-request streaming events, `cancel()`, and `result()`. Thread-safe; either any waiting
  client drives `step()`, or `start()` runs a background loop (what an HTTP server will use).
- `sampling.py` — greedy, temperature, top-k, top-p, repetition penalty. Each request owns its RNG, so seeded
  output does not depend on what else is in the batch.
- `text.py` — `StreamDecoder`: incremental detokenization with stop-string holdback (a stop string split across
  tokens is never partially emitted).
- `scheduler.stats` — counters (prompt/generated tokens, prefill vs decode seconds, TTFT, batch size, cancels,
  timeouts, errors) that Phase 4 will expose as server metrics.

## Verified (`test/gpt/test_phase3_inference.py`)

- Engine logits match the training model to ~1e-6 (tied/untied head, all four activations, stepwise, chunked prefill, batched decode).
- KV-cache greedy generation is token-identical to naive full-sequence recomputation.
- A continuous batch of 4 requests with different prompt and output lengths equals running each alone; seeded sampling likewise.
- Streaming text equals blocking text; stop tokens, stop strings, length and context limits behave.
- Closing a stream, `cancel()`, and timeouts all free the request's KV cache.
- Budget: requests queue instead of exceeding the cache budget; a request that can never fit fails with an error.
- Concurrent streaming clients against the background runner; tampered weights rejected by checksum.
- Speed (tiny model, 64-token prompt, 64 new tokens): 0.08s cached vs 1.49s recomputed (~18x).

## Behaviors worth knowing

- Training never used a BOS token, so none is added. An empty prompt starts from EOS.
- Prompts that don't fit the context raise an error unless `truncate_prompt=True` (keeps the most recent tokens
  and reserves room for generation). Generation stops with `finish_reason="length"` when prompt + completion reaches
  the context length; positions are learned absolute embeddings, so there is no extrapolation past it.
- `finish_reason` is one of `stop`, `length`, `cancelled`, `timeout`, `error`.

## Fixed after release

- `export` used to put the EOS id in the default `stop_token_ids`, so `stop_on_eos=False` did not actually disable EOS stopping (5 of 45 random test models stopped early). The default is now empty; `stop_on_eos` alone controls EOS.

## Honest limits

- CPU/NumPy only. Attention across sequences in a decode step is a Python loop over the batch, so batching helps
  the projections/FFN more than attention; this is the first thing to vectorize (or move to an accelerator backend).
- Eviction: admission control and cancellation exist; there is no automatic preemption of a running request under
  memory pressure and no prefix/shared cache yet.
- Streaming decode re-decodes the full generated text per token (O(n²) in output length; fine at these context sizes).
- `elu`/`selu` were removed from `GPTConfig`: they crash on the autograd `Tensor` (an existing bug in those
  activations), so accepting them would only fail at first forward.
- The wrapped tokenizer is still whitespace word-level BPE; decode normalizes whitespace, so text is not byte-exact.
- Training checkpoints are pickle; exported models are not. Export from checkpoints you trust.
