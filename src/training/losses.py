import numpy as np

from src.core.Tensor import Tensor


def masked_cross_entropy(logits, targets, mask):
    data = logits.data
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    weights = np.asarray(mask, dtype=np.float32).reshape(-1)

    if data.ndim != 2 or data.shape[0] != targets.shape[0] or weights.shape[0] != targets.shape[0]:
        raise ValueError("masked_cross_entropy expects logits (N, V), targets (N,) and mask (N,)")

    count = float(weights.sum())
    denom = max(count, 1.0)

    shifted = data - data.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    rows = np.arange(targets.shape[0])
    nll = log_norm - shifted[rows, targets]

    out = Tensor(np.float32(float(np.dot(nll, weights)) / denom), requires_grad=logits.requires_grad,
                 parents=(logits,), op="MaskedCrossEntropy")
    out.token_count = count

    def _backward():
        if not logits.requires_grad:
            return

        active = np.nonzero(weights)[0]

        if active.size == 0:
            return

        probs = np.exp(shifted[active] - log_norm[active, None])
        probs[np.arange(active.size), targets[active]] -= 1.0
        probs *= (weights[active] * (float(out.grad) / denom))[:, None]
        logits.grad[active] += probs

    out._backward = _backward

    return out
