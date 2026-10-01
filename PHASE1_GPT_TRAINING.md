# PyTensorForge — Phase 1: GPT Training Core

This adds a decoder-only GPT model and a streaming, resumable training
pipeline on top of the existing `Tensor` autograd core, `MultiHeadAttention`,
`AdamW`, and checkpoint utilities. Nothing in `src/core`, `src/neural`,
`src/activations`, `src/initializers`, or `src/loss` was rewritten except two
targeted, backward-compatible fixes described below.

## New packages

- `src/models/gpt/` — `GPTConfig`, `GPTBlock` (pre-norm decoder block), `GPTModel`
  (token + learned positional embeddings, N blocks, final norm, tied or
  untied LM head, cached causal mask).
- `src/tokenization/` — `Tokenizer` abstract interface (`encode`, `decode`,
  `batch_encode`, `encode_stream`, `save`/`load`, `identity`) and
  `PTFBPETokenizer`, a wrapper around the existing `BPETokenizer` that adds
  BOS/EOS/PAD/UNK special tokens, a graceful unknown-token fallback, and a
  fixed `decode()` (the original merged `</w>` markers into other tokens and
  never converted them back to spaces — fixed here).
- `src/data/` — `CorpusIndex` (recursive file discovery, extension filtering,
  optional shuffle, checkpointable file list), `StreamingTextDataset`
  (bounded-buffer file reading → incremental tokenization → token buffer →
  fixed-length sequence packing with EOS at document boundaries →
  batches, with full cursor state for resume), `PrefetchLoader` (bounded
  background-thread prefetch queue).
- `src/training/` — `LRScheduler` (warmup + cosine/linear/constant decay,
  checkpointable), `CheckpointManager` (atomic writes via tempfile +
  `os.replace`, `checkpoint_latest` + `checkpoint_step_N`, rotation via
  `keep_last`/`keep_every`), `Trainer` (gradient accumulation, global-norm
  gradient clipping, SIGINT/SIGTERM-safe checkpointing, JSONL + console
  logging, periodic eval, full resumability).
- `src/config.py` — YAML-backed `TrainConfig` (`model`/`data`/`training`/
  `checkpoint`/`evaluation`/`runtime`).
- `cli.py` — `train`, `resume`, `evaluate`, `inspect`, `tokenize`. `generate`
  and `prepare-dataset` are wired up but intentionally return "not yet
  implemented" (they're KV-cache inference and token-shard preprocessing —
  later phases, not stubbed out silently).

## Two fixes to existing code (both additive, nothing removed)

1. **`Tensor.masked_fill`** now broadcasts the mask to the target shape
   before indexing. Previously a `(1, 1, seq, seq)` mask against
   `(batch, heads, seq, seq)` attention scores raised an `IndexError` —
   this is exactly the shape mismatch causal attention needs, so attention
   was never actually maskable in practice before this fix.
2. **`AdamW`** keyed its per-parameter moment buffers by Python `id(p)`,
   which is not stable across a process restart, so optimizer state could
   never really be checkpointed. It's now keyed by the parameter's position
   in the (stable-ordered) parameter list, and `state_dict`/`load_state_dict`
   were added.

## What streaming/resumability actually guarantees

- Memory use is bounded by `read_buffer_size × workers` for file I/O and by
  `(sequence_length + 1) × micro_batch_size` for the token buffer — not by
  corpus size. An 11 GB or 100 GB corpus is never loaded or tokenized in one
  shot.
- `.jsonl`/`.json` resume to an exact line boundary (byte offset just past
  the last fully-consumed line).
- `.txt` files are treated as one document each; resume for `.txt` is
  file-granular (a file is either untouched or fully consumed), not
  mid-file byte-exact. This is a real, documented limitation, not a silent
  approximation — call it out if you need mid-file resume on huge `.txt`
  files and I'll add a line-chunked mode for that format too.
- A checkpoint captures model weights, optimizer state, scheduler state,
  global step, tokens/examples processed, dataset cursor (file index + byte
  offset + in-flight token buffer), corpus file list, tokenizer identity,
  Python and NumPy RNG state, and full config. Loading with a mismatched
  tokenizer or corpus file list raises immediately instead of silently
  training on the wrong data.

## What's deliberately out of scope for this phase

The `Tensor` core is a single-precision NumPy engine with no device
abstraction. So, honestly:

- **No GPU/accelerator support** — everything here runs on CPU. Training
  speed on a real 11GB+ corpus will be slow; this phase optimizes for
  correctness and bounded memory, not raw throughput.
- **No real mixed precision** — `Trainer` raises immediately if you ask for
  anything but `fp32`, rather than pretending to honor it.
- **No activation checkpointing** — not meaningful without the memory
  pressure a real accelerator has.
- **Prefetching is a single background thread**, not a multi-process worker
  pool — it overlaps I/O/tokenization with the Python-level training step,
  but Python's GIL limits true CPU parallelism here. `data.workers` in the
  config is accepted but not yet wired to a process pool.
- KV-cache inference, model export, the HTTP/SSE API, and the web chat
  client are all later phases (see the original scoping message).

## Try it

```bash
python cli.py tokenize path/to/corpus/ --output tokenizer.json --vocab-size 30000
python cli.py train config.yaml
python cli.py train config.yaml --resume
python cli.py resume checkpoints/checkpoint_step_500.ptf --config config.yaml
python cli.py inspect latest --checkpoint-dir checkpoints/
```

See `config.yaml` fields in `src/config.py` (`ModelSpec`, `DataSpec`,
`TrainingSpec`, `CheckpointSpec`, `EvaluationSpec`, `RuntimeSpec`).
