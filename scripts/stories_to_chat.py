#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys

SUBJECTS = [
    "boy", "girl", "dog", "cat", "rabbit", "bird", "fish", "bear", "duck", "mouse",
    "frog", "monkey", "lion", "elephant", "horse", "pig", "cow", "turtle", "butterfly",
    "princess", "king", "mom", "dad", "teacher", "robot", "dragon", "fairy",
    "toy", "ball", "doll", "car", "boat", "tree", "flower", "garden", "park",
    "beach", "farm", "school", "cake", "balloon", "kite", "bike",
]
SUBJECT_RE = re.compile(r"\b(" + "|".join(SUBJECTS) + r")s?\b", re.I)

SPECIFIC = [
    "Tell me a story about a {s}.",
    "Tell me a short story about a {s}.",
    "Can you tell me a story with a {s} in it?",
    "Write a short story about a {s}.",
    "I want a story about a {s}.",
    "Please tell me a story about a {s}.",
]
GENERIC = [
    "Tell me a story.",
    "Tell me a short story.",
    "Can you tell me a story?",
    "Write me a short story.",
    "I want to hear a story.",
    "Please tell me a story.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-stories", type=int, default=30000)
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    parts = [open(os.path.join(a.out, "train", f"part-{i:04d}.jsonl"), "w", encoding="utf-8") for i in range(a.parts)]
    val = open(os.path.join(a.out, "val.jsonl"), "w", encoding="utf-8")

    paths = []
    for p in a.inputs:
        if os.path.isdir(p):
            paths += [os.path.join(r, n) for r, _, ns in sorted(os.walk(p)) for n in sorted(ns) if n.endswith(".jsonl")]
        else:
            paths.append(p)

    n = {"train": 0, "val": 0, "long": 0, "short": 0, "specific": 0}
    done = False
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    story = json.loads(line).get("text", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    continue
                if len(story) < a.min_chars:
                    n["short"] += 1
                    continue
                if len(story) > a.max_chars:
                    n["long"] += 1
                    continue
                m = SUBJECT_RE.search(story[:200])
                if m:
                    prompt = rng.choice(SPECIFIC).format(s=m.group(1).lower())
                    n["specific"] += 1
                else:
                    prompt = rng.choice(GENERIC)
                rec = json.dumps({"messages": [{"role": "user", "content": prompt},
                                               {"role": "assistant", "content": story}]},
                                 ensure_ascii=False) + "\n"
                if rng.random() < a.val_fraction:
                    val.write(rec); n["val"] += 1
                else:
                    rng.choice(parts).write(rec); n["train"] += 1
                if n["train"] + n["val"] >= a.max_stories:
                    done = True
                    break
        if done:
            break

    for p in parts:
        p.close()
    val.close()
    print(f"train: {n['train']}  val: {n['val']}  with a specific subject: {n['specific']}  "
          f"too long: {n['long']}  too short: {n['short']}")
    if n["train"] == 0:
        sys.exit("no stories written: check the input path and --max-chars")


if __name__ == "__main__":
    main()
