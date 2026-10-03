#!/usr/bin/env python3
import argparse
import json
import os
import sys

SETS = {
    # name: (repo, config, split, kind, fields to keep)
    "tinystories": ("roneneldan/TinyStories", None, "train", "text", ["text"]),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", ["text"]),
    "smoltalk": ("HuggingFaceTB/smoltalk", "all", "train", "chat", ["messages"]),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", "chat", ["instruction", "context", "response"]),
    "no_robots": ("HuggingFaceH4/no_robots", None, "train", "chat", ["messages"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=sorted(SETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-docs", type=int, default=100000)
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("install the Hugging Face library first: pip install datasets")

    repo, config, split, kind, fields = SETS[a.name]
    ds = load_dataset(repo, config, split=split, streaming=True)  # streams: no full download

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}.jsonl")
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in ds:
            rec = {k: row.get(k) for k in fields}
            if kind == "text" and not (rec.get("text") or "").strip():
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n >= a.max_docs:
                break
    print(f"wrote {n} records to {path}")


if __name__ == "__main__":
    main()
