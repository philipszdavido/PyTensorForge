import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pytensorforge.core.Tensor import Tensor
from pytensorforge.inference.config import GenerationConfig
from pytensorforge.inference.export import export_model, flatten_state
from pytensorforge.inference.runtime import load_model
from pytensorforge.inference.scheduler import GenerationRequest
from pytensorforge.models.gpt.config import GPTConfig
from pytensorforge.models.gpt.model import GPTModel
from pytensorforge.tokenization.bpe import PTFBPETokenizer
from pytensorforge.training.checkpoint_manager import CheckpointManager

WORDS = "the quick brown fox jumps over lazy dog cat mat mouse house python code model token stream shard".split()


def make_tokenizer(tmp):
    rng = np.random.default_rng(0)
    text = " ".join(rng.choice(WORDS, 4000))
    tok = PTFBPETokenizer(vocab_size=120)
    tok.fit([text])
    path = os.path.join(tmp, "tok.json")
    tok.save(path)
    return PTFBPETokenizer.load(path), path


def make_model(vocab, tie=True, seed=0, ctx=48):
    cfg = GPTConfig(vocab_size=vocab, context_length=ctx, d_model=32, n_layers=2, n_heads=4, ff_dim=64, tie_weights=tie)
    model = GPTModel(cfg)
    model.build()
    rng = np.random.default_rng(seed)
    for p in model.parameters():
        p.data[:] = p.data + rng.normal(0, 0.4, p.data.shape).astype(np.float32)
    return cfg, model


def save_checkpoint(tmp, cfg, model, tok, name="ckpt"):
    d = os.path.join(tmp, name)
    mgr = CheckpointManager(d)
    mgr.save(1, {
        "framework_version": "test",
        "model_config": cfg.to_dict(),
        "model_state": model.state_dict(),
        "tokenizer_identity": tok.identity,
        "optimizer_steps": 1,
        "tokens_processed": 0,
    })
    return os.path.join(d, "checkpoint_step_1.ptf")


def naive_greedy(model, prompt_ids, n):
    ids = list(prompt_ids)

    for _ in range(n):
        logits = model(Tensor(np.array([ids]), requires_grad=False)).data[0, -1]
        ids.append(int(np.argmax(logits)))

    return ids[len(prompt_ids):]


def main():
    tmp = tempfile.mkdtemp()

    try:
        tok, tok_path = make_tokenizer(tmp)
        V = tok.vocab_size

        for tie in (True, False):
            cfg, model = make_model(V, tie)
            ckpt = save_checkpoint(tmp, cfg, model, tok, f"c{tie}")
            out = os.path.join(tmp, f"m{tie}")
            export_model(ckpt, out, tok_path, GenerationConfig(max_new_tokens=16, do_sample=False, stop_on_eos=False))
            gen = load_model(out)

            prompt = tok.encode("the quick brown fox jumps")
            expected = naive_greedy(model, prompt, 20)
            got = gen.generate(prompt, max_new_tokens=20).token_ids
            assert got == expected, (tie, got, expected)
            print(f"kv-cache greedy == naive recompute (tie_weights={tie})")

        cfg, model = make_model(V, True)
        ckpt = save_checkpoint(tmp, cfg, model, tok)
        out = os.path.join(tmp, "model")
        export_model(ckpt, out, tok_path, GenerationConfig(max_new_tokens=12, do_sample=False, stop_on_eos=False))

        assert sorted(os.listdir(out)) == ["config.json", "generation.json", "tokenizer", "weights"]
        assert not any("optimizer" in k for k in json.load(open(os.path.join(out, "config.json")))["weights"]["tensors"])

        gen = load_model(out)

        seen = []
        prompts = [tok.encode(t) for t in ("the quick", "cat mat mouse house python code", "dog", "lazy dog cat the quick brown fox jumps over")]
        solo = [gen.generate(p, max_new_tokens=n, do_sample=False, stop_on_eos=False).token_ids for p, n in zip(prompts, (10, 7, 15, 5))]

        handles = [gen.submit(GenerationRequest(p, GenerationConfig(max_new_tokens=n, do_sample=False, stop_on_eos=False)))
                   for p, n in zip(prompts, (10, 7, 15, 5))]
        gen.scheduler.run_until_idle()
        batched = [h.result().token_ids for h in handles]
        assert batched == solo, "batched decode differs from solo decode"
        assert gen.scheduler.stats["max_batch_seen"] == 4
        print("heterogeneous continuous batch == solo (greedy), max batch", gen.scheduler.stats["max_batch_seen"])

        def sampled(p, seed):
            return GenerationConfig(max_new_tokens=12, temperature=0.9, top_k=20, top_p=0.9, seed=seed, stop_on_eos=False)

        alone = [gen.generate(p, **{"max_new_tokens": 12, "temperature": 0.9, "top_k": 20, "top_p": 0.9, "seed": 100 + i, "stop_on_eos": False, "do_sample": True}).token_ids
                 for i, p in enumerate(prompts)]
        handles = [gen.submit(GenerationRequest(p, sampled(p, 100 + i))) for i, p in enumerate(prompts)]
        gen.scheduler.run_until_idle()
        together = [h.result().token_ids for h in handles]
        assert alone == together, "seeded sampling depends on batch composition"
        again = gen.generate(prompts[0], max_new_tokens=12, temperature=0.9, top_k=20, top_p=0.9, seed=100, stop_on_eos=False, do_sample=True).token_ids
        assert again == alone[0]
        print("seeded sampling is reproducible and batch-independent")

        res = gen.generate("the quick brown", max_new_tokens=15, do_sample=False, stop_on_eos=False)
        parts = list(gen.generate_stream("the quick brown", max_new_tokens=15, do_sample=False, stop_on_eos=False))
        assert "".join(parts) == res.text and res.usage["completion_tokens"] == 15
        assert res.usage["prompt_tokens"] == len(tok.encode("the quick brown"))
        print("streaming text == blocking text:", repr(res.text[:40]))

        base = gen.generate(prompts[0], max_new_tokens=10, do_sample=False, stop_on_eos=False)
        stop_tok = base.token_ids[3]
        r = gen.generate(prompts[0], max_new_tokens=10, do_sample=False, stop_on_eos=False, stop_token_ids=[stop_tok])
        assert r.finish_reason == "stop" and r.token_ids == base.token_ids[:base.token_ids.index(stop_tok) + 1]
        words = base.text.split()
        if len(words) >= 4:
            marker = words[3]
            r2 = gen.generate(prompts[0], max_new_tokens=10, do_sample=False, stop_on_eos=False, stop_strings=[marker])
            streamed = "".join(gen.generate_stream(prompts[0], max_new_tokens=10, do_sample=False, stop_on_eos=False, stop_strings=[marker]))
            assert marker not in r2.text and r2.finish_reason == "stop" and streamed == r2.text
        r3 = gen.generate(prompts[0], max_new_tokens=3, do_sample=False, stop_on_eos=False)
        assert r3.finish_reason == "length"
        print("stop tokens, stop strings (with holdback), length limit ok")

        it = gen.generate_stream("the quick", events=True, max_new_tokens=40, do_sample=False, stop_on_eos=False)
        next(it)
        assert gen.scheduler.cache_manager.active == 1
        it.close()
        assert gen.scheduler.cache_manager.active == 0 and gen.scheduler.cache_manager.used_bytes == 0
        print("closing a stream cancels generation and frees its kv cache")

        h = gen.submit(GenerationRequest("the quick", GenerationConfig(max_new_tokens=30, do_sample=False, stop_on_eos=False)))
        h.cancel()
        r = h.result()
        assert r.finish_reason == "cancelled"

        r = gen.generate("the quick", max_new_tokens=40, do_sample=False, stop_on_eos=False, timeout_s=0.0000001)
        assert r.finish_reason == "timeout", r.finish_reason
        assert gen.scheduler.cache_manager.used_bytes == 0

        long_prompt = list(np.random.default_rng(1).integers(0, V, 60))
        r = gen.generate(long_prompt, max_new_tokens=5)
        assert r.finish_reason == "error" and "context" in r.error
        r = gen.generate(long_prompt, max_new_tokens=5, truncate_prompt=True, do_sample=False, stop_on_eos=False)
        assert r.finish_reason in ("length", "stop") and r.usage["prompt_tokens"] <= cfg.context_length
        r = gen.generate([V + 5], max_new_tokens=2)
        assert r.finish_reason == "error"
        r = gen.generate("the", max_new_tokens=500, do_sample=False, stop_on_eos=False)
        assert r.finish_reason == "length" and r.usage["total_tokens"] <= cfg.context_length
        print("timeout, cancel, context overflow, invalid ids, context-limit stop ok")

        per_req = gen.engine.new_cache(cfg.context_length).nbytes
        small = load_model(out, cache_budget_bytes=int(per_req * 2.5), max_batch_size=8)
        hs = [small.submit(GenerationRequest(p, GenerationConfig(max_new_tokens=40, do_sample=False, stop_on_eos=False))) for p in prompts]
        peak = 0
        while small.scheduler.step():
            peak = max(peak, small.scheduler.cache_manager.used_bytes)
        assert peak <= int(per_req * 2.5) and all(h.result().finish_reason == "length" for h in hs)
        tiny = load_model(out, cache_budget_bytes=1000)
        assert tiny.generate("the", max_new_tokens=3).finish_reason == "error"
        print("kv cache budget: requests queue, peak", peak, "<= budget", int(per_req * 2.5), "; oversize request rejected")

        gen.start()
        import threading
        results = {}

        def worker(i):
            results[i] = "".join(gen.generate_stream(prompts[i % 4], max_new_tokens=8, do_sample=False, stop_on_eos=False))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        [t.start() for t in threads]
        [t.join(timeout=30) for t in threads]
        gen.stop()
        assert len(results) == 6 and all(results[i] == results[i % 4] for i in range(4, 6))
        print("background runner serves concurrent streaming clients")

        tampered = os.path.join(tmp, "tampered")
        shutil.copytree(out, tampered)
        with open(os.path.join(tampered, "weights", "model.npz"), "r+b") as f:
            f.seek(200)
            f.write(b"\x00\x01\x02")
        try:
            load_model(tampered)
            raise AssertionError("tampered weights loaded")
        except ValueError as exc:
            assert "checksum" in str(exc)
        print("checksum verification rejects tampered weights")

        f16 = os.path.join(tmp, "m16")
        export_model(ckpt, f16, tok_path, dtype="float16")
        size32 = os.path.getsize(os.path.join(out, "weights", "model.npz"))
        size16 = os.path.getsize(os.path.join(f16, "weights", "model.npz"))
        assert size16 < size32 * 0.6
        assert load_model(f16).generate("the quick", max_new_tokens=4, do_sample=False).finish_reason in ("length", "stop")
        print("float16 export", size16, "bytes vs", size32)

        default_gen = json.load(open(os.path.join(out, "generation.json")))
        default_out = os.path.join(tmp, "default_gen")
        export_model(ckpt, default_out, tok_path)
        assert json.load(open(os.path.join(default_out, "generation.json")))["stop_token_ids"] == []

        big_cfg = GPTConfig(vocab_size=V, context_length=256, d_model=128, n_layers=4, n_heads=4, ff_dim=512)
        bm = GPTModel(big_cfg)
        bm.build()
        bckpt = save_checkpoint(tmp, big_cfg, bm, tok, "big")
        bout = os.path.join(tmp, "bigm")
        export_model(bckpt, bout, tok_path)
        bg = load_model(bout)
        prompt = list(np.random.default_rng(2).integers(0, V, 64))
        n = 64
        t0 = time.perf_counter()
        fast = bg.generate(prompt, max_new_tokens=n, do_sample=False, stop_on_eos=False).token_ids
        t_cached = time.perf_counter() - t0
        t0 = time.perf_counter()
        slow = naive_greedy(bm, prompt, n)
        t_naive = time.perf_counter() - t0
        assert fast == slow
        print(f"64 new tokens after 64-token prompt: kv-cache {t_cached:.2f}s vs full recompute {t_naive:.2f}s ({t_naive / t_cached:.1f}x)")

        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
