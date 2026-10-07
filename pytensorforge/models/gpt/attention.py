import threading

import numpy as np

from pytensorforge.core.Tensor import Tensor, matmul_policy
from pytensorforge.neural.Layer import Layer

_mask_cache = {}
_mask_lock = threading.Lock()


def _future_mask(T):
    with _mask_lock:
        mask = _mask_cache.get(T)

        if mask is None:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            _mask_cache[T] = mask

        return mask


def _identity(x):
    return x


def _to_heads(x, B, T, H, hd):
    return x.reshape(B, T, H, hd).transpose(0, 2, 1, 3)


def _from_heads(x, B, T, d):
    return np.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(B, T, d)


def causal_attention(q, k, v, n_heads, rope=None):
    B, T, d = q.shape
    H = n_heads
    hd = d // H

    policy = matmul_policy()
    rnd = _identity if policy is None else policy
    scale = 1.0 / float(np.sqrt(hd))

    qh = _to_heads(q.data, B, T, H, hd)
    kh = _to_heads(k.data, B, T, H, hd)
    vh = _to_heads(v.data, B, T, H, hd)

    cos = sin = None

    if rope is not None:
        cos, sin = rope.tables(np.arange(T))
        qh = rope.rotate(qh, cos, sin)
        kh = rope.rotate(kh, cos, sin)

    qs = rnd(qh)
    ks = rnd(kh)
    vs = rnd(vh)

    scores = rnd(np.matmul(qs, ks.transpose(0, 1, 3, 2)))
    scores *= scale

    if T > 1:
        scores[..., _future_mask(T)] = -np.inf

    scores -= scores.max(axis=-1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=-1, keepdims=True)
    probs = scores

    ps = rnd(probs)
    context = rnd(np.matmul(ps, vs))

    out = Tensor(
        _from_heads(context, B, T, d),
        requires_grad=q.requires_grad or k.requires_grad or v.requires_grad,
        parents=(q, k, v),
        op="CausalAttention",
    )

    def _backward():
        d_out = rnd(_to_heads(out.grad, B, T, H, hd))

        if v.requires_grad:
            d_v = rnd(np.matmul(ps.transpose(0, 1, 3, 2), d_out))
            v.grad += _from_heads(d_v, B, T, d)

        if not (q.requires_grad or k.requires_grad):
            return

        d_p = rnd(np.matmul(d_out, vs.transpose(0, 1, 3, 2)))
        d_s = probs * (d_p - np.sum(d_p * probs, axis=-1, keepdims=True))
        d_s *= scale
        d_s = rnd(d_s)

        if q.requires_grad:
            d_q = rnd(np.matmul(d_s, ks))

            if rope is not None:
                d_q = rope.rotate(d_q, cos, sin, inverse=True)

            q.grad += _from_heads(d_q, B, T, d)

        if k.requires_grad:
            d_k = rnd(np.matmul(d_s.transpose(0, 1, 3, 2), qs))

            if rope is not None:
                d_k = rope.rotate(d_k, cos, sin, inverse=True)

            k.grad += _from_heads(d_k, B, T, d)

    out._backward = _backward

    return out


class CausalSelfAttention(Layer):

    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None

    def build(self, input_shape):
        shape = (self.d_model, self.d_model)
        self.Wq = self.add_weight(shape=shape, initializer="glorot_uniform", name="Wq")
        self.Wk = self.add_weight(shape=shape, initializer="glorot_uniform", name="Wk")
        self.Wv = self.add_weight(shape=shape, initializer="glorot_uniform", name="Wv")
        self.Wo = self.add_weight(shape=shape, initializer="glorot_uniform", name="Wo")
        self.built = True

    def call(self, x, mask=None):
        context = causal_attention(x @ self.Wq, x @ self.Wk, x @ self.Wv, self.num_heads, self.rope)
        return context @ self.Wo

    def state_dict(self):
        return {
            "Wq": self.Wq.data.copy(),
            "Wk": self.Wk.data.copy(),
            "Wv": self.Wv.data.copy(),
            "Wo": self.Wo.data.copy(),
        }

    def load_state_dict(self, state):
        self.Wq.data[:] = state["Wq"]
        self.Wk.data[:] = state["Wk"]
        self.Wv.data[:] = state["Wv"]
        self.Wo.data[:] = state["Wo"]
