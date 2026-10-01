# Phase 7 — Fine-Tuning and Chat Training

Phase 7 adds what turns a pretrained language model into a chat model:

- **`init_from`:** start a new run from existing weights, with a fresh
  optimizer, LR schedule and data position.
- **Chat datasets (`data.format: chat`):** stream JSONL conversations, render
  them with the same chat template the server uses, and learn only the
  assistant replies (loss masking).

It also fixes a reproducibility bug that dates back to Phase 1.

## The chat-model recipe

```bash
pytensorforge tokenize data/pretrain --output tokenizer.json --type bytebpe --vocab-size 8192 \
    --special-token "<|system|>" --special-token "<|user|>" \
    --special-token "<|assistant|>" --special-token "<|end|>"

pytensorforge train pretrain.yaml
pytensorforge train chat.yaml
pytensorforge export checkpoints-chat/checkpoint_latest.ptf --output exports/mygpt --tokenizer tokenizer.json
pytensorforge serve exports/mygpt --name mygpt
```

`chat.yaml` differs from the pretraining config in three places:

```yaml
data:
  train: [data/chat/train]
  validation: [data/chat/val.jsonl]
  tokenizer: tokenizer.json
  format: chat
  chat_template: ptf-chat
  chat_packing: pack
  sequence_length: 512
training:
  init_from: checkpoints/checkpoint_latest.ptf
  total_steps: 2000
  learning_rate: 1.0e-4
  warmup_steps: 50
checkpoint:
  directory: checkpoints-chat
```

The `model:` section must describe the same architecture as the pretrained
model (see "Compatibility" below). Fine-tuning learning rates are usually
about 3–10× lower than pretraining rates.

This replaces the workaround in my earlier advice (appending chat text to the
end of the pretraining corpus). Chat tuning is now a separate run with its own
schedule, and it can use `ptf-chat`. Previously `ptf-chat` was unusable for
training, because special tokens written into a text corpus are encoded as
ordinary text. Chat datasets insert the real control tokens from the template.
`export` picks up the training template automatically, so the served model
uses exactly the format it learned.

## Chat data format

One conversation per line, in `.jsonl` files:

```json
{"messages": [{"role": "system", "content": "Be concise."},
              {"role": "user", "content": "What is a KV cache?"},
              {"role": "assistant", "content": "It stores attention keys and values..."}]}
```

- Roles must be ones the template defines. Both built-in templates support
  `system`, `user` and `assistant`.
- A malformed record stops training with its file and byte position, rather
  than being silently skipped: invalid JSON, a missing `messages` list, a
  non-string `content`, or an unknown role.
- Conversations with no assistant turn teach nothing. They are skipped and
  counted in the dataset state.
- `data.messages_field` changes the key name (default `messages`).

## What is learned

For every assistant message, the model sees the template's generation prompt
(`Assistant:` or `<|assistant|>\n`) exactly as inference renders it, then
learns the reply. The tests check that the tokens before each reply are
**identical** to the prompt the server builds for that turn, for both
templates and across multi-turn history.

| Tokens | Learned? |
|---|---|
| System and user messages | no |
| Generation prompt before each reply | no |
| Assistant reply text and its suffix (including `<|end|>` for `ptf-chat`) | yes |
| EOS after the final reply | yes |

The final EOS is what makes replies stop on their own with the `plain`
template, which has no stop token; the server stops on EOS by default. The
loss is averaged over learned tokens only, and masked tokens receive exactly
zero gradient. The log's `target_tokens` counts learned tokens; `tokens` still
counts all tokens processed.

**Packing:**

- `pack` (default): concatenates conversations into full-length rows, so no
  compute is wasted. A long conversation can be split across rows, so some
  replies are learned without their beginning in view.
- `pad`: one conversation per row, truncated to `sequence_length` and padded
  with EOS (never learned). It keeps every conversation intact but wastes
  compute on padding. Conversations whose learned part is entirely cut off are
  counted as `truncated_away`.

With gradient accumulation, the loss is averaged per micro-batch and then
across micro-batches. Micro-batches with fewer learned tokens therefore weigh
each token slightly more, as in most SFT implementations.

## `init_from`

`training.init_from` accepts a checkpoint file or a run directory (its latest
checkpoint is used). Only the **weights** are taken; the optimizer state, LR
schedule, step counters and data position start fresh. The source is recorded
in every checkpoint of the new run (`init_from`: path, run id, step, context
length).

On `--resume`, an existing checkpoint in the new run's directory takes
precedence and `init_from` is ignored. Interrupting and resuming a fine-tune
is bit-identical to an uninterrupted fine-tune; this is tested through the
CLI.

**Compatibility.** vocab size, d_model, layers, heads, ff_dim, activation,
norm eps, weight tying, position encoding and RoPE theta must match, and so
must the tokenizer. Any mismatch is refused with the differing fields listed.
Only RoPE models may change `context_length`. That makes `init_from` the
fine-tuning step for Phase 6 context extension:

```yaml
model:
  position_encoding: rope
  context_length: 2048
  rope_scaling: linear
  rope_scaling_factor: 4.0
  trained_context_length: 512
data:
  sequence_length: 2048
training:
  init_from: checkpoints-512/checkpoint_latest.ptf
```

## Reproducibility fix

`pytensorforge train` built the model *before* seeding the random number
generators, so each process started from different initial weights even with
the same config and seed. It went unnoticed because every earlier test seeded
manually before building the model. It was found when two identical chat runs
gave different results.

Model construction is now seeded from `training.seed`. A new test runs the
same config twice through the CLI, deliberately disturbing the global RNG
state in between, and requires bit-identical weights. Runs started before this
fix still resume exactly, since the weights come from the checkpoint. A fresh
run of an old config will start from different (now reproducible) initial
weights than it did before.

## Tests

`python -m test.gpt.test_phase7_finetune`:

- **Masked loss:** equals ordinary cross-entropy at full mask (value and
  gradient); averages over unmasked tokens only; gives zero gradient to masked
  tokens; gradient-checked; an all-masked batch is safe.
- **Template alignment:** training contexts equal inference prompts for both
  templates; only replies, stop tokens and the final EOS are learned; templates
  whose generation prompt isn't a prefix of the assistant prefix are refused
  for training.
- **Dataset:** pack and pad layouts, exact mid-stream resume in both modes,
  skip counting, and malformed records reported with location.
- **Fine-tuning:** `init_from` loads weights only; fresh optimizer, schedule
  and data; provenance recorded; CLI resume bit-identical; mismatched
  architectures and tokenizers refused; RoPE context change accepted.
- **Behaviour:** a model tuned for 400 steps on a small synthetic chat task,
  exported and queried through its chat template, answers correctly (5/6 with
  `plain`, 6/6 with `ptf-chat`) and ends 6/6 replies by itself with both
  templates.
- **Reproducibility:** as described above.

## Not included

- **Preference tuning** (RLHF, DPO) is not implemented.
- **Cross-conversation attention masking:** packed rows let a conversation
  attend to the previous one in the same row, as in pretraining.
- **Tool/function-calling roles:** templates define system, user and
  assistant only.
- **`init_from` from an export directory:** it takes training checkpoints only.
