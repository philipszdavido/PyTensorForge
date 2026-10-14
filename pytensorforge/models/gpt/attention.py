import math
import threading

import numpy as _host_np

from pytensorforge.backend import assign, copy_to_numpy, exp_, get_device, xp
from pytensorforge.core.Tensor import Tensor, matmul_policy
from pytensorforge.neural.Layer import Layer

_mask_cache = {}
_mask_lock = threading.Lock()


def _future_mask(T):
    """Additive causal mask: 0 on and below the diagonal, -inf above it. Cached per device."""
    key = (get_device(), T)

    with _mask_lock:
        mask = _mask_cache.get(key)

        if mask is None:
            host = _host_np.where(_host_np.triu(_host_np.ones((T, T), dtype=bool), k=1), -_host_np.inf, 0.0)
            mask = xp.asarray(host.astype(_host_np.float32))
            _mask_cache[key] = mask

        return mask


def _identity(x):
    return x


def _to_heads(x, B, T, H, hd):
    return x.reshape(B, T, H, hd).transpose(0, 2, 1, 3)


def _from_heads(x, B, T, d):
    return xp.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(B, T, d)


def causal_attention(q, k, v, n_heads, rope=None):
    B, T, d = q.shape
    H = n_heads
    hd = d // H

    policy = matmul_policy()
    rnd = _identity if policy is None else policy
    scale = 1.0 / math.sqrt(hd)
    native = None if policy is None else getattr(policy, "native_dtype", None)

    def mm(a, b):
        if native is None:
            return xp.matmul(a, b)

        return xp.matmul(a.astype(native), b.astype(native)).astype(xp.float32)

    qh = _to_heads(q.data, B, T, H, hd)
    kh = _to_heads(k.data, B, T, H, hd)
    vh = _to_heads(v.data, B, T, H, hd)

    cos = sin = None

    if rope is not None:
        cos, sin = rope.slice(0, T)
        qh = rope.rotate(qh, cos, sin)
        kh = rope.rotate(kh, cos, sin)

    qs = rnd(qh)
    ks = rnd(kh)
    vs = rnd(vh)

    scores = rnd(mm(qs, ks.transpose(0, 1, 3, 2)))
    scores *= scale

    if T > 1:
        scores += _future_mask(T)

    scores -= scores.max(axis=-1, keepdims=True)
    scores = exp_(scores)
    scores /= scores.sum(axis=-1, keepdims=True)
    probs = scores

    ps = rnd(probs)
    context = rnd(mm(ps, vs))

    out = Tensor(
        _from_heads(context, B, T, d),
        requires_grad=q.requires_grad or k.requires_grad or v.requires_grad,
        parents=(q, k, v),
        op="CausalAttention",
    )

    def _backward():
        d_out = rnd(_to_heads(out.grad, B, T, H, hd))

        if v.requires_grad:
            d_v = rnd(mm(ps.transpose(0, 1, 3, 2), d_out))
            v.grad += _from_heads(d_v, B, T, d)

        if not (q.requires_grad or k.requires_grad):
            return

        d_p = rnd(mm(d_out, vs.transpose(0, 1, 3, 2)))
        d_s = probs * (d_p - xp.sum(d_p * probs, axis=-1, keepdims=True))
        d_s *= scale
        d_s = rnd(d_s)

        if q.requires_grad:
            d_q = rnd(mm(d_s, ks))

            if rope is not None:
                d_q = rope.rotate(d_q, cos, sin, inverse=True)

            q.grad += _from_heads(d_q, B, T, d)

        if k.requires_grad:
            d_k = rnd(mm(d_s.transpose(0, 1, 3, 2), qs))

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
            "Wq": copy_to_numpy(self.Wq.data),
            "Wk": copy_to_numpy(self.Wk.data),
            "Wv": copy_to_numpy(self.Wv.data),
            "Wo": copy_to_numpy(self.Wo.data),
        }

    def load_state_dict(self, state):
        assign(self.Wq.data, state["Wq"])
        assign(self.Wk.data, state["Wk"])
        assign(self.Wv.data, state["Wv"])
        assign(self.Wo.data, state["Wo"])
