import numpy as np


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample_token(logits, config, rng, seen_ids=None):
    logits = np.array(logits, dtype=np.float64, copy=True)

    penalty = config.repetition_penalty

    if penalty != 1.0 and seen_ids:
        idx = np.fromiter(seen_ids, dtype=np.int64, count=len(seen_ids))
        vals = logits[idx]
        logits[idx] = np.where(vals > 0, vals / penalty, vals * penalty)

    if config.greedy:
        return int(np.argmax(logits))

    logits /= config.temperature

    candidates = None

    if 0 < config.top_k < logits.size:
        candidates = np.argpartition(logits, -config.top_k)[-config.top_k:]
        logits = logits[candidates]

    if config.top_p < 1.0:
        order = np.argsort(-logits, kind="stable")
        probs = _softmax(logits[order])
        cutoff = int(np.searchsorted(np.cumsum(probs), config.top_p)) + 1
        keep = order[:cutoff]
        candidates = keep if candidates is None else candidates[keep]
        logits = logits[keep]

    probs = _softmax(logits)
    choice = int(rng.choice(probs.size, p=probs))

    return int(candidates[choice]) if candidates is not None else choice
