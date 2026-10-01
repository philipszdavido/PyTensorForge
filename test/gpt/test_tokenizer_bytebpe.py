import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.corpus import CorpusIndex
from src.data.document_stream import DocumentReader
from src.data.shard_builder import build_shards
from src.data.validation import validate_corpus, validate_shards
from src.inference.runtime import load_model
from src.inference.text import StreamDecoder
from src.tokenization.base import identity_matches
from src.tokenization.bpe import PTFBPETokenizer
from src.tokenization.bytebpe import ByteLevelBPETokenizer, _bpe_heap, _bpe_simple, train_byte_bpe
from src.tokenization.registry import load_tokenizer

VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "tokenizer", "streaming",
    "café", "naïve", "résumé", "你好", "世界", "日本語", "😀", "🚀", "Привет", "мир",
    "don't", "it's", "x=1", "foo_bar", "12345", "3.14159", "...", "--", "def", "return",
]
SEPS = [" ", " ", " ", "  ", "\n", "\n\n", "\t", "\u00a0", "\u2003", "\r\n"]


def make_text(rng, n):
    return "".join(rng.choice(VOCAB) + rng.choice(SEPS) for _ in range(n))


def main():
    tmp = tempfile.mkdtemp()
    rng = random.Random(7)

    try:
        train = [make_text(rng, 300) for _ in range(60)]
        tok = train_byte_bpe(train, 600, special_tokens=["<|system|>", "<|user|>", "<|assistant|>"], min_frequency=1)

        assert tok.vocab_size == 256 + len(tok.merges) + 6 and len(tok.merges) > 100
        assert train_byte_bpe(train, 300, min_frequency=1).vocab_size == 300
        assert tok.encode("") == [] and tok.encode("a") == [97]

        probes = ["", " ", "\n\n\n", "héllo 你好 😀 Привет", "don't", "x" * 5000, "\x00\x01\x1f\x7f", "a\u2028b", make_text(rng, 400)]
        probes.append(bytes(range(256)).decode("latin-1"))
        for t in probes:
            assert tok.decode(tok.encode(t)) == t, repr(t[:30])
        print("lossless round trip on", len(probes), "strings (control chars, all latin-1, CJK, emoji)")

        srng = random.Random(5)
        for _ in range(400):
            ids = [srng.choice([97, 98, 99, 32, 101, 116, 104]) for _ in range(srng.randint(0, 500))]
            assert _bpe_simple(list(ids), tok._ranks) == _bpe_heap(ids, tok._ranks)
        long = list(("the quick brown fox " * 300).encode())
        assert _bpe_simple(list(long), tok._ranks) == _bpe_heap(long, tok._ranks)
        print("O(n log n) heap merge path == reference path")

        d = os.path.join(tmp, "docs")
        os.makedirs(d)
        cases = 0
        for trial in range(60):
            doc = make_text(rng, rng.randint(5, 500))
            path = os.path.join(d, "f.txt")
            open(path, "w", encoding="utf8").write(doc)
            whole = tok.encode(doc)
            for bs in (64, 129, 500, 4096):
                seg = [i for _, _, t, _ in DocumentReader([path], read_buffer_size=bs).iter_documents() for i in tok.encode(t)]
                assert seg == whole, (trial, bs)
                cases += 1
        os.remove(os.path.join(d, "f.txt"))
        print(f"reader segments encode identically to whole documents ({cases} cases, unicode whitespace included)")

        text = "hi <|user|> there <eos>"
        assert tok.encode(text).count(tok.eos_id) == 0 and tok.decode(tok.encode(text)) == text
        ids = tok.encode(text, allow_special=True)
        assert ids.count(tok.eos_id) == 1 and tok.token_to_id("<|user|>") in ids
        print("special tokens cannot be injected through plain text")

        tpath = os.path.join(tmp, "tok.json")
        tok.save(tpath)
        again = load_tokenizer(tpath)
        assert isinstance(again, ByteLevelBPETokenizer) and again.identity == tok.identity
        assert train_byte_bpe(train, 600, special_tokens=["<|system|>", "<|user|>", "<|assistant|>"], min_frequency=1).merges == tok.merges
        other = train_byte_bpe(train[:5], 600, min_frequency=1)
        assert not identity_matches(tok.identity, other.identity)
        old = {k: v for k, v in tok.identity.items() if k != "fingerprint"}
        assert identity_matches(old, tok.identity)
        print("registry load, deterministic training, fingerprints distinguish same-size tokenizers")

        text = "héllo 你好 😀 wörld"
        ids = tok.encode(text)
        pieces = [tok.token_bytes(i) for i in ids]
        assert any(len(p) and any(b >= 0x80 for b in p) for p in pieces)
        raw = list(text.encode())
        raw_ids = raw
        dec = StreamDecoder(tok)
        out = []
        for i in raw_ids:
            delta = dec.push(i)
            assert "\ufffd" not in delta
            out.append(delta)
        out.append(dec.flush())
        assert "".join(out) == text
        dec = StreamDecoder(tok, stop_strings=["世界"])
        streamed = "".join(dec.push(i) for i in tok.encode("aa 你好 世界 bb")) + dec.flush()
        assert streamed == "aa 你好 " and dec.stopped
        print("streaming decode never emits half a character; stop strings work across multibyte text")

        corpus_dir = os.path.join(tmp, "corpus")
        os.makedirs(corpus_dir)
        for i in range(3):
            open(os.path.join(corpus_dir, f"f{i}.txt"), "w", encoding="utf8").write(make_text(rng, 800))
        with open(os.path.join(corpus_dir, "g.jsonl"), "w", encoding="utf8") as f:
            for _ in range(120):
                f.write(json.dumps({"text": make_text(rng, 20)}, ensure_ascii=rng.random() < 0.5) + "\n")

        r = subprocess.run(
            [sys.executable, "cli.py", "tokenize", corpus_dir, "--output", os.path.join(tmp, "cli_tok.json"),
             "--vocab-size", "700", "--special-token", "<|user|>", "--min-frequency", "1"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        assert r.returncode == 0, r.stderr
        ctok = load_tokenizer(os.path.join(tmp, "cli_tok.json"))
        print("cli tokenize:", r.stdout.strip().split(": ", 1)[1], "| stderr:", r.stderr.strip()[:80] or "-")

        report = validate_corpus(CorpusIndex(corpus_dir), tokenizer=ctok)
        assert report.ok and "unk_rate" not in report.stats
        cpt = report.stats["chars_per_token"]

        wl = PTFBPETokenizer(vocab_size=700)
        wl.fit([make_text(rng, 300) for _ in range(60)])
        wl_report = validate_corpus(CorpusIndex(corpus_dir), tokenizer=wl)
        print(f"chars/token on multilingual sample: byte-level {cpt:.2f} (0 unknowns)  vs  word-level unk_rate {wl_report.stats.get('unk_rate', 0):.1%}")

        shards = os.path.join(tmp, "shards")
        m = build_shards(CorpusIndex(corpus_dir), os.path.join(tmp, "cli_tok.json"), shards, shard_tokens=3000, workers=2, read_buffer_size=257)
        assert validate_shards(shards, ctok, verify_checksums=True).ok
        flat = np.concatenate([np.fromfile(os.path.join(shards, s["file"]), dtype=m["dtype"]) for s in m["shards"]]).astype(np.int64)
        expected = []
        for _, _, t, end in DocumentReader(CorpusIndex(corpus_dir).files, 257).iter_documents():
            expected.extend(ctok.encode(t))
            if end:
                expected.append(ctok.eos_id)
        assert sorted(expected) == sorted(flat.tolist()) and flat.max() < ctok.vocab_size
        print("byte-level shards: multiprocess build matches streaming encode,", m["total_tokens"], "tokens")

        cfg_path = os.path.join(tmp, "cfg.yaml")
        open(cfg_path, "w").write(f"""
model: {{vocab_size: 1, context_length: 32, d_model: 32, n_layers: 2, n_heads: 4, ff_dim: 64}}
data: {{train: [{shards}], tokenizer: {os.path.join(tmp, 'cli_tok.json')}, sequence_length: 32, prefetch: 2}}
training: {{max_tokens: 4000, micro_batch_size: 4, gradient_accumulation_steps: 2, learning_rate: 0.003, warmup_steps: 3, total_steps: 40}}
checkpoint: {{directory: {os.path.join(tmp, 'ckpt')}, interval_steps: 10, keep_last: 2}}
evaluation: {{interval_steps: 0}}
runtime: {{}}
""")
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        r = subprocess.run([sys.executable, "cli.py", "train", cfg_path], capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-800:]
        losses = [json.loads(l)["loss"] for l in open(os.path.join(tmp, "ckpt", "train_log.jsonl")) if '"loss"' in l and '"eval' not in l]
        assert losses[-1] < losses[0], losses
        r = subprocess.run([sys.executable, "cli.py", "resume", "latest", "--config", cfg_path], capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-800:]
        r = subprocess.run(
            [sys.executable, "cli.py", "export", os.path.join(tmp, "ckpt", "checkpoint_latest.ptf"),
             "--tokenizer", os.path.join(tmp, "cli_tok.json"), "--output", os.path.join(tmp, "model")],
            capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-800:]
        print(f"train->resume->export with byte-level tokenizer: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

        gen = load_model(os.path.join(tmp, "model"))
        prompt = "héllo 你好 😀 the quick"
        res = gen.generate(prompt, max_new_tokens=25, do_sample=False, stop_on_eos=False)
        assert res.usage["prompt_tokens"] == len(ctok.encode(prompt))
        streamed = "".join(gen.generate_stream(prompt, max_new_tokens=25, do_sample=False, stop_on_eos=False))
        assert streamed == res.text
        assert res.text == ctok.decode(res.token_ids)
        print("generation with unicode prompt; streamed == blocking == tokenizer.decode(ids):", repr(res.text[:36]))

        big = "".join(make_text(random.Random(9), 4000) for _ in range(1))
        ids = tok.encode(big)
        t0 = time.perf_counter()
        for _ in range(3):
            tok._cache.clear()
            tok.encode(big)
        cold = len(big.encode()) * 3 / (time.perf_counter() - t0) / 1e6
        t0 = time.perf_counter()
        for _ in range(3):
            tok.encode(big)
        warm = len(big.encode()) * 3 / (time.perf_counter() - t0) / 1e6
        print(f"encode throughput (one process): cold cache {cold:.1f} MB/s, warm cache {warm:.1f} MB/s; {len(big.encode()) / len(ids):.2f} bytes/token")

        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
