import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.Tensor import Tensor, no_grad
from src.loss.CrossEntropyWithLogitsLoss import CrossEntropyWithLogitsLoss
from src.models.gpt.config import GPTConfig
from src.models.gpt.context import EXTENSION_METHODS, extend_context
from src.models.gpt.model import GPTModel
from src.optimizers.AdamW import AdamW
from src.tokenization.bytebpe import train_byte_bpe


def english_corpus():
    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    paths = sorted(glob.glob("/usr/share/common-licenses/*")) + sorted(glob.glob(os.path.join(repo, "*.md")))
    texts = []

    for p in paths:
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())

    return texts


def train(model, ids, context, steps, batch, lr, seed):
    rng = np.random.default_rng(seed)
    opt = AdamW(lr=lr, weight_decay=0.01)
    loss_fn = CrossEntropyWithLogitsLoss()
    params = model.parameters()
    vocab = model.config.vocab_size
    losses = []

    for step in range(steps):
        starts = rng.integers(0, len(ids) - context - 1, batch)
        x = np.stack([ids[s:s + context] for s in starts])
        y = np.stack([ids[s + 1:s + context + 1] for s in starts])

        for p in params:
            p.grad = None

        logits = model(Tensor(x, requires_grad=False))
        loss = loss_fn(logits.reshape(-1, vocab), Tensor(y, requires_grad=False).reshape(-1))
        loss.backward(release=True)
        opt.lr = lr * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        opt.step(params)
        losses.append(float(loss.data))

    return losses


def position_losses(model, ids, length, windows, seed):
    rng = np.random.default_rng(seed)
    vocab = model.config.vocab_size
    total = np.zeros(length)

    with no_grad():
        for _ in range(windows):
            s = int(rng.integers(0, len(ids) - length - 1))
            x = ids[s:s + length][None]
            y = ids[s + 1:s + length + 1]
            logits = model(Tensor(x, requires_grad=False)).data[0].astype(np.float64)
            logits -= logits.max(axis=-1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
            total += -logp[np.arange(length), y]

    return total / windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-context", type=int, default=64)
    parser.add_argument("--eval-context", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--windows", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    texts = english_corpus()
    split = int(0.9 * len(texts))
    tok = train_byte_bpe(texts[:split], vocab_size=768)
    train_ids = np.array(tok.encode("\n\n".join(texts[:split])), dtype=np.int64)
    held_ids = np.array(tok.encode("\n\n".join(texts[split:])), dtype=np.int64)
    print(f"corpus: {len(train_ids):,} train tokens, {len(held_ids):,} held-out tokens, vocab {tok.vocab_size}")

    base = GPTConfig(vocab_size=tok.vocab_size, context_length=args.train_context, d_model=args.d_model,
                     n_layers=args.layers, n_heads=args.heads, ff_dim=4 * args.d_model, position_encoding="rope")
    np.random.seed(args.seed)
    model = GPTModel(base)
    model.build()
    start = time.perf_counter()
    losses = train(model, train_ids, args.train_context, args.steps, args.batch, 3e-3, args.seed)
    print(f"trained {args.steps} steps at context {args.train_context} in {time.perf_counter() - start:.0f}s: "
          f"loss {np.mean(losses[:20]):.3f} -> {np.mean(losses[-20:]):.3f}")

    state = model.state_dict()
    T, L = args.train_context, args.eval_context
    in_range = position_losses(model, held_ids, T, args.windows, args.seed + 1)
    print(f"\nheld-out loss at trained length {T}: {in_range.mean():.3f}")

    buckets = [(0, T)]
    edge = T

    while edge < L:
        buckets.append((edge, min(L, edge * 2)))
        edge *= 2

    header = "method        " + "  ".join(f"{a:>4}-{b - 1:<4}" for a, b in buckets) + "   all"
    print(f"\nheld-out loss by position at length {L} (lower is better)")
    print(header)

    for method in EXTENSION_METHODS:
        extended = GPTModel(extend_context(base, L, method))
        extended.build()
        extended.load_state_dict(state)
        per_pos = position_losses(extended, held_ids, L, args.windows, args.seed + 1)
        cells = "  ".join(f"{per_pos[a:b].mean():9.3f}" for a, b in buckets)
        print(f"{method:<12}  {cells}  {per_pos.mean():6.3f}")


if __name__ == "__main__":
    main()
