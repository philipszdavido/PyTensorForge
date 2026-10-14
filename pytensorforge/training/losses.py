import numpy as _host_np

from pytensorforge.backend import to_device, to_numpy, xp
from pytensorforge.core.Tensor import Tensor


def masked_cross_entropy(logits, targets, mask):
    data = logits.data
    targets = xp.asarray(targets, dtype=xp.int64).reshape(-1)
    weights = xp.asarray(mask, dtype=xp.float32).reshape(-1)

    if data.ndim != 2 or data.shape[0] != targets.shape[0] or weights.shape[0] != targets.shape[0]:
        raise ValueError("masked_cross_entropy expects logits (N, V), targets (N,) and mask (N,)")

    active_host = _host_np.nonzero(to_numpy(mask).reshape(-1))[0]
    count = float(weights.sum())
    denom = max(count, 1.0)

    shifted = data - data.max(axis=1, keepdims=True)
    log_norm = xp.log(xp.exp(shifted).sum(axis=1))
    rows = xp.arange(targets.shape[0])
    nll = log_norm - shifted[rows, targets]

    out = Tensor(_host_np.float32(float((nll * weights).sum()) / denom), requires_grad=logits.requires_grad,
                 parents=(logits,), op="MaskedCrossEntropy")
    out.token_count = count

    def _backward():
        if not logits.requires_grad:
            return

        if active_host.size == 0:
            return

        active = to_device(active_host)

        probs = xp.exp(shifted[active] - log_norm[active, None])
        probs[xp.arange(active.size), targets[active]] -= 1.0
        probs *= (weights[active] * (float(out.grad) / denom))[:, None]
        logits.grad[active] += probs

    out._backward = _backward

    return out
