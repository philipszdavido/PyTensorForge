# PyTensorForge — Phase 2: Data Pipeline, Shards, Validation, Experiments

## What's new

- **Token shards (Mode B)** — `prepare-dataset` streams a raw corpus through the
  tokenizer and writes `shard-wNNN-NNNNN.bin` files (uint16 when the vocab fits,
  else uint32) plus `manifest.json` (token counts, dtype, tokenizer identity,
  vocab size, per-shard SHA-256, corpus version, source files). Shards are
  written atomically. `--workers N` uses N spawned processes (real CPU
  parallelism for tokenization); each worker owns a contiguous slice of the
  file list and writes its own shards, so no worker holds more than one shard
  buffer plus one document.
- **`ShardedTokenDataset`** — streams shards with `np.fromfile` at an offset,
  packs `(seq_len + 1)` blocks, and resumes at exact token granularity
  (shard index + token offset + carry). Point `data.train` at a shard
  directory and the CLI selects it automatically.
- **Dataset validation** — `validate-dataset PATH [--tokenizer T] [--checksums]`.
  Raw corpora: file existence/readability, bounded-sample scan for malformed
  JSON, bad UTF-8, empty docs, and unknown-token rate. Shards: manifest,
  sizes vs token counts, tokenizer identity and vocab compatibility, optional
  full checksum verification, source-corpus drift. Nothing scans the full
  corpus unless `--checksums` is passed.
- **Deterministic train/val split** — `data.val_split_fraction` partitions raw
  corpora at file level by a seeded hash of the file name (no shuffling of
  content, nothing loaded).
- **Experiment tracking** — `experiment.json` in the checkpoint directory:
  run ID, full config, tokenizer identity, hardware, seed, resume history, and
  final metrics including the profiler summary.
- **Profiling hooks** — every step's JSONL record includes seconds spent in
  `data_wait`, `host_prep`, `forward`, `backward`, `optimizer`, `checkpoint`.
  A large `data_wait` share means the device is starved by the CPU pipeline.
- **Training loop additions** — epochs, `max_epochs`, ETA and tokens
  remaining, max RSS, evaluation every N steps or every N tokens,
  `pause()`/`request_stop()`, `--no-save-on-exit`.

## Bugs found and fixed in Phase 1 code

1. **Prefetch + checkpoint skipped data.** With prefetch enabled the dataset
   cursor had advanced past batches still sitting in the queue, so a resume
   silently skipped them. Each batch now carries the cursor snapshot taken
   right after it was produced; checkpoints store the cursor of the last
   batch actually consumed.
2. **`.txt` files were read whole into RAM**, violating the bounded-memory
   requirement. They now stream in whitespace-aligned segments, with exact
   byte-offset resume mid-file (this removes the Phase 1 "file-granular"
   limitation).
3. **Evaluation could only run once**: the validation cursor was never reset.
4. Prefetch worker threads could linger on shutdown; they now stop cleanly.

## Verification (`test/gpt/test_phase2_pipeline.py`)

- Mode A and Mode B produce the identical token multiset.
- Shard and text datasets resume to exactly the same batch sequence as an
  uninterrupted pass.
- A trainer interrupted at step 3 and resumed ends with the same weights as an
  uninterrupted 6-step run, with prefetch off and on.
- Repeated evaluation is deterministic.

## Honest limits

- Shard order across workers follows worker slices, not global file order;
  training-time order is deterministic but not identical to raw-file order.
- Mode B does not shuffle within shards; shuffle at the file level before
  building, or add shard-level shuffling later.
- Prefetch remains a single thread (tokenization for training-time Mode A is
  GIL-bound); use Mode B for repeated runs, which moves tokenization out of the
  training loop entirely.
- Still no GPU, fp16/bf16, or activation checkpointing (the Tensor core is a
  float32 NumPy engine); the trainer rejects those settings explicitly.
- Tokenizer *fitting* still holds a bounded sample (`--max-chars`) in memory,
  and the wrapped BPE is word-level (whitespace split, lowercase by default),
  which is a poor fit for code or multilingual data.
- Checkpoints are pickle files: only load ones you created.
