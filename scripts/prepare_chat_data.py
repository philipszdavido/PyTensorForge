#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys

ROLE_MAP = {"system": "system", "developer": "system", "user": "user", "human": "user",
            "assistant": "assistant", "gpt": "assistant", "bot": "assistant"}


def files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in sorted(os.walk(p)):
                for n in sorted(names):
                    if n.endswith(".jsonl"):
                        yield os.path.join(root, n)
        elif os.path.isfile(p):
            yield p
        else:
            print(f"warning: {p} not found", file=sys.stderr)


def text(x):
    return x.replace("\r\n", "\n").strip() if isinstance(x, str) else ""


def to_messages(obj):
    if isinstance(obj.get("messages"), list):
        raw = [(m.get("role"), m.get("content")) for m in obj["messages"] if isinstance(m, dict)]
    elif isinstance(obj.get("conversations"), list):
        raw = [(m.get("from"), m.get("value")) for m in obj["conversations"] if isinstance(m, dict)]
    elif "instruction" in obj and ("output" in obj or "response" in obj):
        prompt = text(obj.get("instruction"))
        extra = text(obj.get("input") or obj.get("context"))
        if extra:
            prompt = f"{prompt}\n\n{extra}"
        raw = [("user", prompt), ("assistant", obj.get("output") or obj.get("response"))]
    elif "prompt" in obj and "response" in obj:
        raw = [("user", obj["prompt"]), ("assistant", obj["response"])]
    else:
        return None
    out = []
    for role, content in raw:
        role = ROLE_MAP.get(str(role).lower())
        content = text(content)
        if role and content:
            out.append({"role": role, "content": content})

    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out if any(m["role"] == "assistant" for m in out) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    parts = [open(os.path.join(a.out, "train", f"part-{i:04d}.jsonl"), "w", encoding="utf-8") for i in range(a.parts)]
    val = open(os.path.join(a.out, "val.jsonl"), "w", encoding="utf-8")
    seen, n = set(), {"train": 0, "val": 0, "skipped": 0, "dupes": 0, "long": 0}

    for path in files(a.inputs):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    n["skipped"] += 1
                    continue
                msgs = to_messages(obj) if isinstance(obj, dict) else None
                if not msgs:
                    n["skipped"] += 1
                    continue
                if sum(len(m["content"]) for m in msgs) > a.max_chars:
                    n["long"] += 1
                    continue
                key = hash(json.dumps(msgs, sort_keys=True))
                if key in seen:
                    n["dupes"] += 1
                    continue
                seen.add(key)
                rec = json.dumps({"messages": msgs}, ensure_ascii=False) + "\n"
                if rng.random() < a.val_fraction:
                    val.write(rec); n["val"] += 1
                else:
                    rng.choice(parts).write(rec); n["train"] += 1

    for p in parts:
        p.close()
    val.close()
    print(f"train: {n['train']}  val: {n['val']}  skipped: {n['skipped']}  duplicates: {n['dupes']}  too long: {n['long']}")
    if n["val"] == 0:
        print("note: validation file is empty; raise --val-fraction or remove data.validation from chat.yaml")


if __name__ == "__main__":
    main()
