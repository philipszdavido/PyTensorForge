# PyTensorForge — Byte-level BPE tokenizer

`src/tokenization/bytebpe.py` adds `ByteLevelBPETokenizer` (implements the existing `Tokenizer` interface) and a trainer.
`python cli.py tokenize CORPUS --output tok.json --vocab-size 32000` now trains it by default (`--type word` keeps the old one).

## Properties
- **Lossless.** The base vocabulary is the 256 byte values, so any text round-trips exactly (control chars, emoji, CJK,
  invalid-looking input) and there is no unknown token. The old word-level tokenizer lowercased, normalized whitespace and
  turned anything unseen into `<unk>`.
- **Ids:** `0..255` bytes, then one id per merge, then special tokens (`<bos> <eos> <pad>` plus any `--special-token`,
  e.g. chat markers for Phase 4). `vocab_size` is the real total; the model reads it from the tokenizer.
- **Pre-tokenization:** GPT-2 style regex (contractions, letters, digits, punctuation, whitespace runs, leading space
  attached to the next word), implemented with `re` (no extra dependency), so Unicode letter/number classes are approximations of `\p{L}`/`\p{N}`.
- **Special tokens are inert in ordinary text** (`"<eos>"` typed by a user or present in the corpus is just characters).
  Use `encode(text, allow_special=True)` when a chat template deliberately inserts them.
- **Integrity:** the file stores a fingerprint of merges+specials; loading a modified file fails. `identity` includes the
  fingerprint, so two different tokenizers of the same size are no longer treated as compatible by checkpoints, shards,
  export or `load_model`. (`identity_matches` compares shared keys only, so older artifacts without a fingerprint still load, with the older, weaker check. The word-level tokenizer now writes a fingerprint too.)

## Encoding
Cached per pretoken; merges applied by lowest rank, with an O(n log n) linked-list/heap path for long pretokens
(differentially tested against the reference loop). Streaming encode is exact: `DocumentReader` now cuts `.txt` streams
only at the start of a whitespace run whose previous character is not whitespace, which is always a pretoken boundary, so
segment-wise encoding equals whole-document encoding (320+ fuzz cases including NBSP/EM-space/LS). Streaming *decode*
(`StreamDecoder`) uses an incremental UTF-8 decoder, so a character split across tokens is held back rather than emitted
half-formed.

## Trainer
Streams text (`--max-chars` sampled evenly across files, per-file budget), counts pretokens with a bounded table
(`--max-unique-words`, lossy pruning of rare entries when exceeded), then standard BPE with an incremental pair-count
heap. Deterministic. Stops early (with a warning) if no pair reaches `--min-frequency`.

## Numbers from this sandbox (single process, pure Python; treat as rough)
- Encode ~8-10 MB/s per process on synthetic text; ~3.2 bytes/token on the multilingual test corpus, ~6.9 on a synthetic English-like one.
- Training on a 4.5 MB synthetic sample to 8k vocab took ~1.3 s. Real corpora have far more distinct words, so expect training on hundreds of MB to take minutes to tens of minutes.
- `prepare-dataset --workers N` parallelizes encoding across files. At ~8 MB/s per process, 11 GB is about 23 minutes on one worker (11,000 MB / 8 MB/s ≈ 1,375 s), divided by the worker count if disk keeps up. That is a one-time cost, and it is an extrapolation from a synthetic benchmark, not a measurement on your data.

## Honest limits
- If a file has no whitespace for `4 x read_buffer_size` bytes (default 4 MB), the reader hard-cuts at a UTF-8 boundary and
  encoding across that seam can differ from whole-document encoding. Whitespace-free scripts (Chinese/Japanese prose without spaces)
  are safe only while runs stay under that size; a single enormous JSONL document is encoded in one piece (memory ∝ document size).
- Training samples the beginning of each file; it does not random-access large files.
- Casing/normalization: none (no NFC). Pretokenizer digits are not length-limited.
- Existing shards/checkpoints made with the word-level tokenizer are not compatible with a new tokenizer; re-run `prepare-dataset`.
