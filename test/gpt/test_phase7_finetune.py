import argparse
import json
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cli
from pytensorforge.config import TrainConfig
from pytensorforge.core.Tensor import Tensor
from pytensorforge.data.chat_dataset import ChatRecordError, ChatSFTDataset
from pytensorforge.data.corpus import CorpusIndex
from pytensorforge.inference.chat_template import ChatTemplate, ChatTemplateError
from pytensorforge.inference.export import export_model, flatten_state
from pytensorforge.inference.runtime import load_model
from pytensorforge.loss.CrossEntropyWithLogitsLoss import CrossEntropyWithLogitsLoss
from pytensorforge.tokenization.bytebpe import train_byte_bpe
from pytensorforge.tokenization.registry import load_tokenizer
from pytensorforge.training.checkpoint_manager import read_checkpoint
from pytensorforge.training.losses import masked_cross_entropy
from pytensorforge.training.trainer import Trainer
from test.gpt.test_phase5_training import check, directional_check

SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
COLORS = ["red", "blue", "green", "black", "white", "pink"]
FILLER = "please tell me kindly about this thing now".split()


def conversation(rng, color):
    noise = " ".join(rng.choice(FILLER) for _ in range(rng.randint(2, 6)))
    return {"messages": [{"role": "user", "content": f"{noise} color {color}"},
                         {"role": "assistant", "content": f"{color} is a color."}]}


def write_chat(path, n, seed, extra=None):
    rng = random.Random(seed)

    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps(conversation(rng, rng.choice(COLORS))) + "\n")

            if extra and i in extra:
                f.write(extra[i] + "\n")


def make_tokenizer(tmp):
    rng = random.Random(0)
    texts = [" ".join(rng.choice(FILLER + COLORS + ["color", "is", "a", "color."]) for _ in range(40))
             for _ in range(200)]
    texts.append(open(os.path.join(os.path.dirname(__file__), "..", "..", "PHASE4_SERVING.md")).read())
    tok = train_byte_bpe(texts, vocab_size=420, special_tokens=SPECIALS)
    path = os.path.join(tmp, "tok.json")
    tok.save(path)
    return load_tokenizer(path), path


def write_config(tmp, name, tok_path, data, model=None, training=None):
    cfg = {
        "model": {"vocab_size": 0, "context_length": 32, "d_model": 48, "n_layers": 2, "n_heads": 4, "ff_dim": 96,
                  "position_encoding": "rope", **(model or {})},
        "data": {"tokenizer": tok_path, "sequence_length": 32, "prefetch": 0, **data},
        "training": {"total_steps": 20, "micro_batch_size": 8, "gradient_accumulation_steps": 1,
                     "learning_rate": 3e-3, "warmup_steps": 5, "decay": "cosine", "max_epochs": 50,
                     **(training or {})},
        "checkpoint": {"directory": os.path.join(tmp, name), "interval_steps": 1000, "keep_last": 2},
        "evaluation": {"interval_steps": 0},
    }
    path = os.path.join(tmp, f"{name}.yaml")

    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)

    return path


def trainer_from(path, seed=1):
    config = TrainConfig.load(path)
    tokenizer = load_tokenizer(config.data.tokenizer)
    random.seed(seed)
    np.random.seed(seed)
    model = cli._build_model(config, tokenizer)
    train, val = cli._build_datasets(config, tokenizer)
    return Trainer(model, train, config, tokenizer, val_dataset=val, install_signal_handlers=False), model


def run_cli_train(path, resume=False):
    return cli.cmd_train(argparse.Namespace(config=path, resume=resume, no_save_on_exit=False))


def flat(model):
    return np.concatenate([p.data.ravel() for p in model.parameters()])


def test_masked_loss():
    rng = np.random.default_rng(0)
    logits_data = rng.standard_normal((12, 7)).astype(np.float32)
    targets = rng.integers(0, 7, 12)

    a = Tensor(logits_data.copy(), requires_grad=True)
    ref = CrossEntropyWithLogitsLoss()(a, Tensor(targets, requires_grad=False))
    ref.backward()
    b = Tensor(logits_data.copy(), requires_grad=True)
    got = masked_cross_entropy(b, targets, np.ones(12))
    got.backward()
    check(abs(float(ref.data) - float(got.data)) < 1e-5 and np.allclose(a.grad, b.grad, atol=1e-6),
          "full mask equals ordinary cross-entropy (value and gradient)")

    mask = (rng.random(12) > 0.5).astype(np.float32)
    active = np.nonzero(mask)[0]
    c = Tensor(logits_data[active].copy(), requires_grad=True)
    sub = CrossEntropyWithLogitsLoss()(c, Tensor(targets[active], requires_grad=False))
    d = Tensor(logits_data.copy(), requires_grad=True)
    masked = masked_cross_entropy(d, targets, mask)
    masked.backward()
    sub.backward()
    check(abs(float(sub.data) - float(masked.data)) < 1e-5, "masked loss averages over unmasked tokens only")
    check(np.allclose(d.grad[active], c.grad, atol=1e-6) and not d.grad[mask == 0].any(),
          "masked tokens receive exactly zero gradient")
    check(masked.token_count == mask.sum(), "loss reports its token count")

    err = directional_check(lambda t: masked_cross_entropy(t, targets, mask) * 1.0,
                            [Tensor(logits_data.copy(), requires_grad=True)])
    check(err < 1e-2, f"masked loss gradient check ({err:.1e})")

    e = Tensor(logits_data.copy(), requires_grad=True)
    empty = masked_cross_entropy(e, targets, np.zeros(12))
    empty.backward()
    check(float(empty.data) == 0.0 and not e.grad.any(), "an all-masked batch contributes nothing")
    print("masked loss: equals CE at full mask, ignores masked tokens exactly, gradient-checked, empty-safe")


def test_training_tokens(tok):
    conv = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi, color red?"},
            {"role": "assistant", "content": "red is a color."}, {"role": "user", "content": "And blue?"},
            {"role": "assistant", "content": "blue is a color."}]

    for name in ("plain", "ptf-chat"):
        bound = ChatTemplate.builtin(name).bind(tok)
        ids, learn = bound.training_tokens(conv)
        check(len(ids) == len(learn), f"{name}: one flag per token")

        for k in (2, 4):
            prompt = bound.render(conv[:k]).token_ids
            check(ids[:len(prompt)] == prompt, f"{name}: training context equals the inference prompt for turn {k}")

        first_prompt = bound.render(conv[:2]).token_ids
        check(not any(learn[:len(first_prompt)]), f"{name}: nothing before the first reply is learned")

        spans = []
        i = 0

        while i < len(learn):
            if learn[i]:
                j = i
                while j < len(learn) and learn[j]:
                    j += 1
                spans.append(ids[i:j])
                i = j
            else:
                i += 1

        check(len(spans) == 2, f"{name}: exactly the two assistant replies are learned")
        check("blue is a color." in tok.decode(spans[1]) and "red is a color." in tok.decode(spans[0]),
              f"{name}: learned spans are the replies")
        check(spans[1][-1] == tok.eos_id, f"{name}: EOS after the final reply is learned, so generation stops")
        check("Be brief" not in tok.decode([t for t, l in zip(ids, learn) if l]), f"{name}: system text not learned")

        if name == "ptf-chat":
            end = tok.special_tokens["<|end|>"]
            check(all(end in s for s in spans), "ptf-chat: the <|end|> stop token is learned in every reply")

    broken = ChatTemplate.from_dict({**ChatTemplate.builtin("plain").to_dict(), "name": "odd",
                                     "generation_prompt": ["Bot:"]}).bind(tok)

    try:
        broken.training_tokens(conv)
        check(False, "a generation prompt that is not the assistant prefix is rejected for training")
    except ChatTemplateError:
        pass

    print("chat training tokens: contexts equal inference prompts, only replies (+stop, +EOS) learned, both templates")


def test_chat_dataset(tmp, tok):
    d = os.path.join(tmp, "chatdata")
    os.makedirs(d)
    write_chat(os.path.join(d, "a.jsonl"), 60, 1,
               extra={5: "", 9: json.dumps({"messages": [{"role": "user", "content": "no reply here"}]})})
    write_chat(os.path.join(d, "b.jsonl"), 40, 2)
    bound = ChatTemplate.builtin("plain").bind(tok)

    for packing in ("pack", "pad"):
        def run(limit=None, state=None):
            ds = ChatSFTDataset(CorpusIndex(d), bound, 24, packing=packing)

            if state is not None:
                ds.load_state_dict(json.loads(json.dumps(state)))

            out = []
            gen = ds.batches(3)

            for batch in gen:
                out.append(batch)

                if limit and len(out) >= limit:
                    break

            gen.close()
            return out, ds

        full, ds = run()
        check(ds.records == 100 and ds.skipped == 1, f"{packing}: counts records and skips reply-less ones")
        x, y, m = full[0]
        check(x.shape == y.shape == m.shape and np.array_equal(x[:, 1:], y[:, :-1]), f"{packing}: aligned shapes")
        check(0 < m.mean() < 1, f"{packing}: masks cover part of each row")

        head, ds_mid = run(limit=4)
        tail, _ = run(state=ds_mid.state_dict())
        joined = head + tail
        check(len(joined) == len(full) and all(all(np.array_equal(p, q) for p, q in zip(a, b))
                                               for a, b in zip(joined, full)),
              f"{packing}: resume from a mid-stream state is exact")

    one = os.path.join(tmp, "one")
    os.makedirs(one)

    with open(os.path.join(one, "c.jsonl"), "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "hi"},
                                         {"role": "assistant", "content": "red is a color."}]}) + "\n")

    ids, learn = bound.training_tokens([{"role": "user", "content": "hi"},
                                        {"role": "assistant", "content": "red is a color."}])
    x, y, m = next(ChatSFTDataset(CorpusIndex(one), bound, 24, packing="pad").batches(1))
    n = len(ids) - 1
    check(np.array_equal(x[0, :n], ids[:-1]) and np.array_equal(m[0, :n], learn[1:]),
          "pad: the row is the conversation with its mask shifted onto targets")
    check(not m[0, n:].any() and (x[0, n:] == tok.eos_id).all(), "pad: padding is EOS and never learned")

    for bad_line, why in (("{not json", "invalid JSON"),
                          (json.dumps({"messages": [{"role": "robot", "content": "x"}]}), "role 'robot'"),
                          (json.dumps({"text": "hello"}), "'messages' list")):
        bad = os.path.join(tmp, f"bad{abs(hash(why))}")
        os.makedirs(bad)
        write_chat(os.path.join(bad, "x.jsonl"), 3, 3, extra={1: bad_line})

        try:
            list(ChatSFTDataset(CorpusIndex(bad), bound, 24).batches(2))
            check(False, f"malformed record accepted ({why})")
        except ChatRecordError as exc:
            check(why in str(exc) and "x.jsonl" in str(exc) and "byte" in str(exc), f"error locates the record ({why})")

    try:
        ChatSFTDataset(CorpusIndex([os.path.join(tmp, "tok.json")]), bound, 24)
        check(False, "non-jsonl chat files rejected")
    except ValueError:
        pass

    print("chat dataset: pack and pad, exact mid-stream resume, skip counting, malformed records located")


def test_finetune_from_pretrained(tmp, tok, tok_path):
    text_dir = os.path.join(tmp, "text")
    os.makedirs(text_dir)
    rng = random.Random(4)

    with open(os.path.join(text_dir, "t.txt"), "w") as f:
        for _ in range(400):
            f.write(" ".join(rng.choice(FILLER + COLORS) for _ in range(14)) + "\n")

    pre = write_config(tmp, "pre", tok_path, {"train": [text_dir], "sequence_length": 16},
                       model={"context_length": 16}, training={"total_steps": 8})
    check(run_cli_train(pre) == 0, "pretraining run")
    pre_ckpt = os.path.join(tmp, "pre")
    pre_payload = read_checkpoint(pre_ckpt)

    chat_dir = os.path.join(tmp, "chat")
    os.makedirs(chat_dir)
    write_chat(os.path.join(chat_dir, "train.jsonl"), 400, 11)
    write_chat(os.path.join(tmp, "val.jsonl"), 40, 12)

    sft = write_config(tmp, "sft", tok_path, {"train": [chat_dir], "validation": [os.path.join(tmp, "val.jsonl")],
                                              "format": "chat", "chat_template": "plain"},
                       model={"context_length": 32, "rope_scaling": "linear", "rope_scaling_factor": 2.0,
                              "trained_context_length": 16},
                       training={"init_from": pre_ckpt, "total_steps": 12})

    t, m = trainer_from(sft)
    t.initialize_from(pre_ckpt)
    loaded = flatten_state(m.state_dict())
    source = flatten_state(pre_payload["model_state"])
    check(loaded.keys() == source.keys() and all(np.array_equal(loaded[k], source[k]) for k in loaded),
          "init_from loads the pretrained weights")
    check(t.optimizer_steps == 0 and t.tokens_processed == 0 and t.scheduler.state_dict() != pre_payload["scheduler_state"],
          "optimizer, schedule and counters start fresh")
    check(t.init_source["optimizer_steps"] == 8 and t.init_source["context_length"] == 16, "provenance recorded")

    ref_dir = os.path.join(tmp, "sft_ref")
    ref = sft.replace("sft.yaml", "sft_ref.yaml")
    with open(sft) as f:
        c = yaml.safe_load(f)
    c["checkpoint"]["directory"] = ref_dir
    with open(ref, "w") as f:
        yaml.safe_dump(c, f)
    check(run_cli_train(ref) == 0, "uninterrupted fine-tune")

    t1, _ = trainer_from(sft)
    t1.initialize_from(pre_ckpt)
    log_step = t1._log

    def stop_early(*a):
        log_step(*a)
        if t1.optimizer_steps >= 5:
            t1.request_stop()

    t1._log = stop_early
    t1.train()
    check(t1.optimizer_steps == 5, "fine-tune interrupted")
    check(run_cli_train(sft, resume=True) == 0, "fine-tune resumed through the CLI")

    a = read_checkpoint(ref_dir)
    b = read_checkpoint(os.path.join(tmp, "sft"))
    check(a["optimizer_steps"] == b["optimizer_steps"] == 12, "both runs complete")
    check(all(np.array_equal(x, y) for x, y in zip(_leaves(a["model_state"]), _leaves(b["model_state"]))),
          "resumed fine-tune is bit-identical: resume restores the checkpoint instead of re-initializing")
    check(b["init_from"]["path"] == os.path.abspath(pre_ckpt) and b["target_tokens"] < b["tokens_processed"],
          "checkpoint records provenance and counts only learned tokens as targets")

    for bad_model, why in (({"d_model": 64}, "d_model"),
                           ({"position_encoding": "learned", "context_length": 32}, "position_encoding")):
        bad = write_config(tmp, f"bad_{why}", tok_path, {"train": [chat_dir], "format": "chat"}, model=bad_model)
        tb, _ = trainer_from(bad)

        try:
            tb.initialize_from(pre_ckpt)
            check(False, f"incompatible init_from accepted ({why})")
        except ValueError as exc:
            check(why in str(exc), f"incompatibility explained ({why})")

    other_tok = train_byte_bpe(["completely different text"] * 20, vocab_size=420, special_tokens=SPECIALS)
    other_path = os.path.join(tmp, "other_tok.json")
    other_tok.save(other_path)
    bad = write_config(tmp, "bad_tok", other_path, {"train": [chat_dir], "format": "chat"})
    tb, _ = trainer_from(bad)

    try:
        tb.initialize_from(pre_ckpt)
        check(False, "init_from with a different tokenizer accepted")
    except ValueError as exc:
        check("tokenizer" in str(exc), "tokenizer mismatch explained")

    print("init_from: loads weights only, fresh optimizer/schedule/data, provenance, CLI resume bit-identical, "
          "incompatible architectures and tokenizers refused, rope context change allowed")
    return sft


def _leaves(state):
    if isinstance(state, dict):
        for k in sorted(state):
            yield from _leaves(state[k])
    elif isinstance(state, (list, tuple)):
        for v in state:
            yield from _leaves(v)
    else:
        yield np.asarray(state)


def test_chat_behaviour(tmp, tok, tok_path):
    chat_dir = os.path.join(tmp, "behave")
    os.makedirs(chat_dir)
    write_chat(os.path.join(chat_dir, "train.jsonl"), 600, 21)
    results = {}

    for template in ("plain", "ptf-chat"):
        cfg = write_config(tmp, f"behave_{template}", tok_path,
                           {"train": [chat_dir], "format": "chat", "chat_template": template},
                           training={"total_steps": 400, "learning_rate": 1e-2, "warmup_steps": 10})
        check(run_cli_train(cfg) == 0, f"{template}: chat tuning run")
        ckpt_dir = os.path.join(tmp, f"behave_{template}")
        export_dir = os.path.join(tmp, f"export_{template}")
        export_model(read_checkpoint_path(ckpt_dir), export_dir, tok_path)

        with open(os.path.join(export_dir, "config.json")) as f:
            check(json.load(f)["chat_template"]["name"] == template, f"{template}: export defaults to the training template")

        gen = load_model(export_dir, max_batch_size=4)
        correct = 0
        stopped = 0

        try:
            bound = gen.bound_chat_template()

            for color in COLORS:
                prompt = bound.render([{"role": "user", "content": f"please tell me color {color}"}]).token_ids
                result = gen.generate(prompt, max_new_tokens=16, temperature=0.0,
                                      stop_token_ids=bound.stop_token_ids, stop_strings=bound.stop_strings)
                text = result.text.strip()
                correct += text.startswith(f"{color} is a color")
                stopped += result.finish_reason == "stop"
        finally:
            gen.stop()

        results[template] = (correct, stopped)
        check(correct >= 5, f"{template}: the tuned model answers correctly ({correct}/6)")
        check(stopped >= 5, f"{template}: replies end by themselves instead of running to max_tokens ({stopped}/6)")

    print("chat behaviour: " + ", ".join(f"{k} {c}/6 correct, {s}/6 stopped on their own" for k, (c, s) in results.items()))


def test_cli_training_is_reproducible(tmp, tok_path):
    chat_dir = os.path.join(tmp, "repro")
    os.makedirs(chat_dir)
    write_chat(os.path.join(chat_dir, "train.jsonl"), 80, 31)
    states = []

    for run in range(2):
        cfg = write_config(tmp, f"repro{run}", tok_path, {"train": [chat_dir], "format": "chat"},
                           training={"total_steps": 6})
        np.random.seed(1000 + run)
        random.seed(2000 + run)
        np.random.random(17 * (run + 1))
        check(run_cli_train(cfg) == 0, "reproducibility run")
        states.append(list(_leaves(read_checkpoint(os.path.join(tmp, f"repro{run}"))["model_state"])))

    check(all(np.array_equal(a, b) for a, b in zip(*states)),
          "two CLI runs of the same config give bit-identical weights, whatever the prior RNG state")
    print("reproducibility: CLI training seeds model initialization from the config")


def read_checkpoint_path(directory):
    return os.path.join(directory, "checkpoint_latest.ptf")


def main():
    tmp = tempfile.mkdtemp()

    try:
        tok, tok_path = make_tokenizer(tmp)
        test_masked_loss()
        test_training_tokens(tok)
        test_chat_dataset(tmp, tok)
        test_cli_training_is_reproducible(tmp, tok_path)
        test_finetune_from_pretrained(tmp, tok, tok_path)
        test_chat_behaviour(tmp, tok, tok_path)
        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
