import argparse
import gc
import os
import random
import shutil
import sys
import tempfile
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pytensorforge.core.Tensor import Tensor
from pytensorforge.data.corpus import CorpusIndex
from pytensorforge.data.streaming_dataset import StreamingTextDataset
from pytensorforge.inference.engine import InferenceModel
from pytensorforge.inference.export import flatten_state
from pytensorforge.loss.CrossEntropyWithLogitsLoss import CrossEntropyWithLogitsLoss
from pytensorforge.models.gpt.config import GPTConfig
from pytensorforge.models.gpt.model import GPTModel
from pytensorforge.tokenization.bytebpe import train_byte_bpe
from pytensorforge.tokenization.registry import load_tokenizer
from pytensorforge.training.precision import autocast


def bench_step(args):
    cfg = GPTConfig(vocab_size=512, context_length=args.seq, d_model=args.d_model, n_layers=args.layers,
                    n_heads=4, ff_dim=4 * args.d_model, position_encoding=args.position_encoding)
    np.random.seed(0)
    model = GPTModel(cfg)
    model.build()
    loss_fn = CrossEntropyWithLogitsLoss()
    rng = np.random.default_rng(0)
    x = rng.integers(0, 512, (args.batch, args.seq))
    y = rng.integers(0, 512, (args.batch, args.seq))

    for checkpointing in (False, True):
        model.activation_checkpointing = checkpointing

        for precision in ("fp32", "bf16"):
            def step():
                for p in model.parameters():
                    p.grad = None

                with autocast(precision):
                    logits = model(Tensor(x, requires_grad=False))
                    b, s, v = logits.shape
                    loss = loss_fn(logits.reshape(b * s, v), Tensor(y, requires_grad=False).reshape(b * s))
                loss.backward(release=True)

            step()
            gc.collect()
            start = time.perf_counter()

            for _ in range(args.repeats):
                step()

            elapsed = (time.perf_counter() - start) / args.repeats
            tracemalloc.start()
            step()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f"step  checkpointing={str(checkpointing):5s} precision={precision}: "
                  f"{elapsed:.3f}s/step  peak {peak / 2**20:.1f} MiB  "
                  f"{args.batch * args.seq / elapsed:.0f} tok/s")


def bench_decode(args):
    cfg = GPTConfig(vocab_size=2000, context_length=1024, d_model=args.d_model, n_layers=args.layers, n_heads=4,
                    ff_dim=4 * args.d_model, position_encoding=args.position_encoding)
    np.random.seed(0)
    model = GPTModel(cfg)
    model.build()
    engine = InferenceModel(cfg, flatten_state(model.state_dict()))
    rng = np.random.default_rng(0)

    for batch in (1, 8, 16):
        caches = []

        for b in range(batch):
            cache = engine.new_cache(1024)
            engine.prefill(rng.integers(0, 2000, args.prompt + 7 * b), cache)
            caches.append(cache)

        ids = rng.integers(0, 2000, batch)
        engine.decode_batch(ids, caches)
        start = time.perf_counter()

        for _ in range(args.decode_steps):
            engine.decode_batch(ids, caches)

        elapsed = (time.perf_counter() - start) / args.decode_steps
        print(f"decode batch={batch:2d} context~{args.prompt}: {elapsed * 1e3:.2f} ms/step  {batch / elapsed:.0f} tok/s")


def bench_tokenize(args):
    words = "stream shard token model train resume checkpoint cursor batch epoch loss gradient".split()
    rng = random.Random(0)
    tmp = tempfile.mkdtemp()

    try:
        for i in range(args.files):
            with open(os.path.join(tmp, f"f{i}.txt"), "w") as f:
                for _ in range(args.lines):
                    f.write(" ".join(rng.choice(words) for _ in range(40)) + "\n")

        size = sum(os.path.getsize(os.path.join(tmp, n)) for n in os.listdir(tmp))

        with open(os.path.join(tmp, "f0.txt")) as f:
            tok = train_byte_bpe([f.read()[:150000]], vocab_size=1000)

        tok_path = os.path.join(tmp, "tok.json")
        tok.save(tok_path)
        tok = load_tokenizer(tok_path)

        for workers in sorted({1, *args.workers}):
            ds = StreamingTextDataset(CorpusIndex(tmp), tok, 256, workers=workers)
            start = time.perf_counter()
            tokens = sum(x.size for x, _ in ds.batches(16))
            elapsed = time.perf_counter() - start
            print(f"tokenize workers={workers}: {size / 2**20 / elapsed:.2f} MB/s ({tokens} tokens in {elapsed:.2f}s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["step", "decode", "tokenize", "all"], default="all")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--position-encoding", default="learned")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--files", type=int, default=4)
    parser.add_argument("--lines", type=int, default=2500)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    print(f"cpus available: {os.cpu_count()}")

    if args.what in ("step", "all"):
        bench_step(args)

    if args.what in ("decode", "all"):
        bench_decode(args)

    if args.what in ("tokenize", "all"):
        bench_tokenize(args)


if __name__ == "__main__":
    main()
