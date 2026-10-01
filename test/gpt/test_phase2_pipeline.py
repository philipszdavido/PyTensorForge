import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import CheckpointSpec, DataSpec, EvaluationSpec, ModelSpec, RuntimeSpec, TrainConfig, TrainingSpec
from src.data.corpus import CorpusIndex
from src.data.shard_builder import build_shards
from src.data.sharded_dataset import ShardedTokenDataset
from src.data.streaming_dataset import StreamingTextDataset
from src.data.validation import validate_corpus, validate_shards
from src.models.gpt.config import GPTConfig
from src.models.gpt.model import GPTModel
from src.tokenization.bpe import PTFBPETokenizer
from src.training.trainer import Trainer

WORDS = "the quick brown fox jumps over lazy dog cat mat mouse house python code model token stream shard".split()


def make_corpus(root, seed=0):
    rng = random.Random(seed)
    os.makedirs(root)

    for i in range(3):
        with open(os.path.join(root, f"f{i}.txt"), "w") as f:
            for _ in range(400):
                f.write(" ".join(rng.choice(WORDS) for _ in range(12)) + "\n")

    for i in range(2):
        with open(os.path.join(root, f"g{i}.jsonl"), "w") as f:
            for _ in range(150):
                f.write(json.dumps({"text": " ".join(rng.choice(WORDS) for _ in range(20))}) + "\n")


def collect(ds, bs, limit=None):
    out = []

    for x, y in ds.batches(bs):
        out.append(x.copy())

        if limit and len(out) >= limit:
            break

    return out


def main():
    tmp = tempfile.mkdtemp()

    try:
        corpus_dir = os.path.join(tmp, "corpus")
        make_corpus(corpus_dir)

        texts = []
        for p in CorpusIndex(corpus_dir).files:
            with open(p) as f:
                texts.append(f.read()[:5000])

        tok = PTFBPETokenizer(vocab_size=120)
        tok.fit(texts)
        tok_path = os.path.join(tmp, "tok.json")
        tok.save(tok_path)
        tok = PTFBPETokenizer.load(tok_path)

        corpus = CorpusIndex(corpus_dir)
        seq = 16

        text_ds = StreamingTextDataset(corpus, tok, seq, read_buffer_size=257)
        a = np.concatenate(collect(text_ds, 4))

        shard_dir = os.path.join(tmp, "shards")
        manifest = build_shards(corpus, tok_path, shard_dir, shard_tokens=5000, workers=2, read_buffer_size=257)

        assert manifest["total_tokens"] > 0
        assert len(manifest["shards"]) > 1

        report = validate_shards(shard_dir, tok, verify_checksums=True)
        assert report.ok, report.to_dict()

        flat_shards = []
        for s in manifest["shards"]:
            flat_shards.append(np.fromfile(os.path.join(shard_dir, s["file"]), dtype=manifest["dtype"]))
        flat_shards = np.concatenate(flat_shards).astype(np.int64)

        flat_text = []
        ds2 = StreamingTextDataset(CorpusIndex(corpus_dir), tok, 10 ** 9, read_buffer_size=257)
        for _, _, text, end in ds2.reader.iter_documents():
            if text:
                flat_text.extend(tok.encode(text))
            if end:
                flat_text.append(tok.eos_id)

        assert sorted(flat_text) == sorted(flat_shards.tolist()), "shard token multiset differs from streaming tokens"
        print("mode A/B token multiset identical:", len(flat_text))

        shard_ds = ShardedTokenDataset(shard_dir, seq, tok)
        full = collect(shard_ds, 4)
        assert sum(b.shape[0] for b in full) > 4

        shard_ds = ShardedTokenDataset(shard_dir, seq, tok)
        it = shard_ds.batches(4)
        first = [next(it)[0].copy() for _ in range(3)]
        state = json.loads(json.dumps(shard_ds.state_dict()))
        resumed = ShardedTokenDataset(shard_dir, seq, tok)
        resumed.load_state_dict(state)
        rest = collect(resumed, 4)
        joined = np.concatenate(first + rest)
        assert np.array_equal(joined, np.concatenate(full)), "shard resume not exact"
        print("shard dataset exact resume ok")

        text_ds = StreamingTextDataset(corpus, tok, seq, read_buffer_size=257)
        it = text_ds.batches(4)
        first = [next(it)[0].copy() for _ in range(5)]
        state = json.loads(json.dumps(text_ds.state_dict()))
        resumed = StreamingTextDataset(corpus, tok, seq, read_buffer_size=257)
        resumed.load_state_dict(state)
        rest = collect(resumed, 4)
        allb = collect(StreamingTextDataset(corpus, tok, seq, read_buffer_size=257), 4)
        assert np.array_equal(np.concatenate(first + rest), np.concatenate(allb)), "text resume not exact"
        print("text dataset exact resume ok")

        rep = validate_corpus(corpus, tokenizer=tok)
        assert rep.ok, rep.to_dict()

        for prefetch in (0, 3):
            run_resume_test(tmp, tok, shard_dir, prefetch)

        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_config(tok, ckpt, prefetch, total_steps):
    return TrainConfig(
        model=ModelSpec(vocab_size=tok.vocab_size, context_length=16, d_model=16, n_layers=1, n_heads=2, ff_dim=32),
        data=DataSpec(sequence_length=16, prefetch=prefetch),
        training=TrainingSpec(
            total_steps=total_steps, micro_batch_size=2, gradient_accumulation_steps=2,
            learning_rate=1e-3, warmup_steps=2, decay="cosine", max_epochs=5,
        ),
        checkpoint=CheckpointSpec(directory=ckpt, interval_steps=1000, keep_last=2),
        evaluation=EvaluationSpec(interval_steps=0),
        runtime=RuntimeSpec(),
    )


def build_trainer(tok, shard_dir, ckpt, prefetch, total_steps, val=False):
    cfg = make_config(tok, ckpt, prefetch, total_steps)
    model = GPTModel(GPTConfig(vocab_size=tok.vocab_size, context_length=16, d_model=16, n_layers=1, n_heads=2, ff_dim=32))
    ds = ShardedTokenDataset(shard_dir, 16, tok)
    vd = ShardedTokenDataset(shard_dir, 16, tok) if val else None
    random.seed(1)
    np.random.seed(1)
    model.build()
    return Trainer(model, ds, cfg, tok, val_dataset=vd, install_signal_handlers=False), model


def flat(model):
    return np.concatenate([p.data.ravel() for p in model.parameters()])


def run_resume_test(tmp, tok, shard_dir, prefetch):
    np.random.seed(7)
    ref_dir = os.path.join(tmp, f"ref{prefetch}")
    t_ref, m_ref = build_trainer(tok, shard_dir, ref_dir, prefetch, 6)
    state0 = m_ref.state_dict()
    t_ref.train()

    run_dir = os.path.join(tmp, f"run{prefetch}")
    t1, m1 = build_trainer(tok, shard_dir, run_dir, prefetch, 3)
    m1.load_state_dict(state0)
    t1.train()

    t2, m2 = build_trainer(tok, shard_dir, run_dir, prefetch, 6)
    assert t2.load_checkpoint()
    assert t2.optimizer_steps == 3
    t2.train()

    assert t2.optimizer_steps == t_ref.optimizer_steps == 6
    assert t2.tokens_processed == t_ref.tokens_processed
    assert np.allclose(flat(m2), flat(m_ref), atol=1e-6), f"resumed weights diverge (prefetch={prefetch})"
    print(f"trainer resume matches uninterrupted run (prefetch={prefetch})")

    tv, _ = build_trainer(tok, shard_dir, os.path.join(tmp, f"ev{prefetch}"), 0, 2, val=True)
    result = tv.evaluate()
    assert result is not None
    again = tv.evaluate()
    assert again is not None and abs(again[0] - result[0]) < 1e-9
    print("repeatable evaluation ok")


if __name__ == "__main__":
    main()
