# Phase 5 — RoPE, Faster Autograd, Activation Checkpointing, Mixed Precision, Parallel Tokenization

Phase 5 closes the training-side gaps left open by Phases 1–4: rotary position
encoding, activation checkpointing, reduced-precision numerics with dynamic
loss scaling, and multi-process tokenization for raw-text (Mode A) training.
Along the way, profiling uncovered three inefficiencies in the autograd core
that roughly halved step time once fixed.

Every new option defaults to the previous behaviour. Existing configs,
checkpoints and exported models keep working unchanged.

## Configuration

```yaml
model:
  position_encoding: rope      # learned (default) | rope
  rope_theta: 10000.0

data:
  workers: 4                   # tokenizer processes for raw-text training; 1 = in-process

training:
  precision: bf16              # fp32 (default) | bf16 | fp16
  loss_scaling: auto           # auto (dynamic for fp16, off otherwise) | dynamic | none
  initial_loss_scale: 65536.0
  loss_scale_growth_interval: 2000
  activation_checkpointing: true
```

## Measured results

Absolute CPU timings in the development sandbox varied by about 3× between
sessions, so every comparison below was measured back-to-back in one session.
The before/after pair alternated Phase 4 and Phase 5 code three times.
Reproduce the Phase 5 side with `python -m test.gpt.bench_phase5`.

Training step, GPT with d_model 128, 4 layers, batch 8 × 128 tokens:

| | Phase 4 | Phase 5 |
|---|---|---|
| Step time (median of 3 interleaved rounds) | 0.42 s | 0.19 s (about 2.2× faster) |
| Peak traced memory | 286 MB | 178 MB (−38%) |

Same model, Phase 5 options (single session, 5 repeats):

| Setting | Step time | Peak memory |
|---|---|---|
| fp32 | 0.19 s | 110 MiB |
| fp32 + activation checkpointing | 0.16 s | 47 MiB |
| bf16 emulation | 0.35 s | 149 MiB |
| bf16 emulation + activation checkpointing | 0.25 s | 47 MiB |

How to read these:

- **Activation checkpointing** cut peak memory to 43% here (33% on a 6-layer
  model with dropout in the test suite). It was also slightly *faster*,
  probably because the much smaller working set reduces fresh-allocation and
  cache costs. Do not rely on that. Recomputation normally costs about one
  extra forward pass per block, and the test suite measured it ~10% slower on
  a different shape.
- **bf16/fp16 emulation costs** roughly 1.8× the step time and ~35% more memory
  on this backend. The rounded operand copies are kept for backward. Use it to
  validate reduced-precision training behaviour, not for speed.
- **Parallel tokenization.** This sandbox has a single CPU core, so no speedup
  is possible here. On that one core, two workers were slower than in-process
  tokenization (3.5 vs 16.4 MB/s on a small corpus) because of process startup
  and IPC. Correctness (identical tokens and resume state) is verified; the
  speedup on multi-core hardware is **not** measured. Run
  `python -m test.gpt.bench_phase5 --what tokenize --workers 2 4 8` on your
  machine and keep `data.workers: 1` if it doesn't help.

## Autograd core fixes

Profiling a training step showed three problems:

- **Eager gradient buffers.** Every intermediate tensor allocated a
  `zeros_like` gradient at creation, about 28% of step time and roughly double the
  activation memory. Gradients are now allocated lazily on first write.
- **GELU running in float64.** `np.sqrt(2 / np.pi)` is an `np.float64` scalar.
  Under NumPy 2's promotion rules it silently promoted the whole activation to
  float64. The constant is now a Python float, so GELU stays in float32.
- **Dropout at p = 0** still drew a random mask per call. It now returns its
  input unchanged.

Also added:

- `no_grad()`: a thread-local context in which operations record no graph.
  Used by evaluation and activation checkpointing.
- `backward(release=True)`: after each node's backward runs, its closure,
  parent references and gradient buffer are dropped. Memory is then freed as
  backward proceeds instead of waiting for the cyclic garbage collector. The
  trainer always uses it.
- An iterative topological sort, so very deep graphs no longer hit Python's
  recursion limit.

## Fused causal attention

`src/models/gpt/attention.py` computes masked causal attention as one
operation with a hand-written backward pass. It replaces a chain of about ten
autograd nodes, each holding a T×T intermediate and its gradient.

On identical weights it matches the previous `MultiHeadAttention` in values and
gradients, so learned-position models train as before. The test suite checks
its gradients numerically with and without RoPE and verifies causality.
Parameter names are unchanged (`attn.Wq/Wk/Wv/Wo`), so checkpoints are
compatible.

## Rotary position encoding (RoPE)

With `position_encoding: rope`, queries and keys are rotated by
position-dependent angles (rotate-half convention, frequencies
`theta^(-2i/head_dim)`). The model then has no learned position table.

- One implementation (`src/models/gpt/rope.py`) serves training and the NumPy
  inference engine.
- The engine rotates keys **before** storing them in the KV cache. Cached keys
  never need re-rotation, and batched decode handles sequences at different
  positions.
- Tests confirm that attention scores depend only on relative position, that
  the rotation preserves norms and is exactly invertible, and that engine
  logits (chunked prefill and mixed-position batched decode) match the training
  model to within 2e-4.
- Export, `load_model`, `generate` and the HTTP server work unchanged.
- The head dimension must be even. Invalid settings are rejected at config time.
- Loading a learned-position checkpoint into a RoPE model, or the reverse, is
  refused.

RoPE does not by itself extend usable context beyond `context_length`;
positions past the table raise an error. Context extension (extrapolate,
linear, NTK) was added in Phase 6; see `PHASE6_CONTEXT_EXTENSION.md`.

## Activation checkpointing

With `activation_checkpointing: true`, each transformer block runs its forward
pass under `no_grad` and keeps only its input. During backward the block is
recomputed with the graph enabled, back-propagated, and released. This trades
one extra forward pass per block for not holding every block's intermediates
at once.

Correctness properties, all tested:

- Dropout masks are **replayed exactly**: the RNG state is captured before the
  first pass and restored for the recomputation.
- The global random-number stream is left undisturbed afterwards.
- Gradients are **bit-identical** to the non-checkpointed path.
- A full training run, with dropout, produces the same loss curve.
- Interrupted runs resume bit-identically.

This is a runtime setting, not part of the architecture. A checkpoint trained
with it enabled can be resumed with it disabled, and vice versa.

## Mixed precision

`precision: bf16 | fp16` rounds matrix-multiply operands, outputs and their
gradients to the target format, as GPU autocast does. Weights, optimizer state,
layer norm, softmax and the loss stay in fp32. bf16 rounding is
round-to-nearest-even on the float32 bit pattern; fp16 uses NumPy's float16.

**What this is and is not.** On the NumPy CPU backend it reproduces
reduced-precision *numerics*, but it makes training slower and heavier
(see Measured results), because arrays are still stored as float32 and the
rounded copies are extra work. The trainer prints this at
startup. It is useful for checking that a model and its hyperparameters train
stably in reduced precision. Its machinery (master weights, dynamic loss
scaling, overflow skipping, checkpointable scaler state) is what an accelerator
backend would reuse.

**Dynamic loss scaling** (on by default for fp16):

- The loss is multiplied by the scale before backward.
- If any gradient is non-finite, the step is skipped: no optimizer or LR
  scheduler update. The skip is logged (`"skipped": "non_finite_gradients"`)
  and the scale is halved.
- After `loss_scale_growth_interval` clean steps the scale doubles.
- The scale and counters are saved in checkpoints and restored on resume.
- The JSONL log records `precision`, `loss_scale` and `skipped_steps`.
- Expected overflow warnings from NumPy are silenced only while loss scaling is
  active.

Tests cover rounding edge cases (ties, inf, NaN, overflow), scoped autocast,
scaler backoff, growth and state round-trips, and bit-identical resume for bf16
and fp16. A forced-overflow run skips steps, backs off and recovers. bf16
changes the weights measurably but only slightly relative to fp32.

## Parallel tokenization (Mode A)

With `data.workers > 1`, raw-text training tokenizes documents in a pool of
worker processes (`src/data/parallel_encode.py`):

- The main process still reads documents in order. Consecutive documents are
  grouped into batches of about 256 KB and sent to workers.
- At most `2 × workers` batches are in flight, so memory stays bounded.
- Results are consumed strictly in submission order. Tokens, batches and the
  dataset resume cursor are identical to single-process tokenization, and
  resuming from a mid-stream checkpoint is exact. Both are tested with the
  word-level and byte-level BPE tokenizers.
- The pool is terminated when the batch iterator is closed (end of epoch,
  resume, shutdown). The tests check that no worker processes are left behind.
- Validation datasets always tokenize in-process.

**The `__main__` guard is required.** Workers use the `spawn` start method,
which re-imports your main module in each worker. If you drive training from
your own Python script rather than the `pytensorforge` CLI, the script must
protect its entry point:

```python
if __name__ == "__main__":
    main()
```

Without the guard, every worker tries to start its own pool and the run fails
with a `RuntimeError` about the bootstrapping phase. The CLI is already
guarded.

For repeated runs over the same corpus, Mode B (`prepare-dataset` token shards)
is still preferable: it tokenizes once instead of every epoch.

## Inference engine: float64 leak fixed

The NumPy inference engine had the same GELU promotion bug as training, with
worse consequences. The float64 activation fed the residual stream, so from
the first layer onward **every matmul in prefill and decode ran in float64**,
and the engine (and the Phase 4 server) returned float64 logits. It had done
so since Phase 3. Existing tests missed it because they compared values within
a tolerance, never dtypes.

After the fix the engine is float32 end to end. A new test checks every
activation with both position encodings. Greedy outputs still match full
recomputation exactly.

Decode throughput, same session, before → after
(`python -m test.gpt.bench_phase5 --what decode`):

| Model | Batch | Context | Before | After |
|---|---|---|---|---|
| d128, 4 layers | 8 | ~64 | 1,403 tok/s | 4,591 tok/s (3.3×) |
| d256, 6 layers | 8 | ~256 | 293 tok/s | 925 tok/s (3.2×) |
| d256, 6 layers | 16 | ~256 | 329 tok/s | 1,139 tok/s (3.5×) |
| d512, 8 layers | 8 | ~512 | 69 tok/s | 186 tok/s (2.7×) |

The Phase 3 KV-cache test now reports a 15× speedup over full recomputation,
up from about 5×.

### Why decode attention still loops over sequences

After the fix, per-sequence attention is 24–36% of a decode step, so I
measured the obvious alternative before changing it. Copying each sequence's
cache into a padded batch and doing one masked batched attention was
**slower** than the loop in 3 of 4 shapes (for example 0.81 vs 0.42 ms per layer
at batch 8, context 256). The copy costs more than the Python overhead it
removes.

An idealized pooled cache (all sequences already in one padded array, no copy)
would save about 4% of a batch-8 step and 16% at batch 16. It would also
require fixed cache slots, which conflicts with the per-request byte budgets
the KV-cache manager enforces. I kept the loop. Revisit this if batch sizes of
16+ become the common case.

## Other fixes

- **Evaluation ran with dropout active.** `evaluate()` now switches the model
  to inference mode (`GPTModel.set_training(False)`), runs under `no_grad`, and
  restores both afterwards.
- **Checkpoint compatibility.** Architecture comparison now normalizes the
  stored model config with current defaults. Checkpoints written before Phase 5
  (without `position_encoding`/`rope_theta`) resume normally. Genuine mismatches
  are still refused.
- `FRAMEWORK_VERSION` is now `0.3.0-phase5`.

## Tests

```bash
python -m test.gpt.test_phase5_training
python -m test.gpt.bench_phase5 --what all
```

All earlier suites (Phases 2–4, tokenizer, browser) pass after these changes.

The Phase 4 browser test had a latent race, found during this regression run.
It clicked Stop as soon as the assistant bubble had any text, but the "…"
typing indicator counts as text. A Stop landing before the first token
correctly produces a "stopped before any output" error instead of a stopped
message, so the test failed about 1 run in 4. It now waits for real streamed
content and passed 10/10 consecutive runs. The client itself was correct.

## Remaining limits

- **CPU only.** There is still no GPU backend; `device` accepts `cpu`/`auto`.
  Mixed precision is numerics-only for this reason.
- **Decode attention** in the inference engine still loops over sequences in
  a batch. This was measured and kept deliberately; see above.
- **Checkpointing granularity** is whole transformer blocks only.
- **Parallel tokenization** needs the `__main__` guard in user scripts, and its
  speedup is unmeasured on multi-core hardware in this environment.
