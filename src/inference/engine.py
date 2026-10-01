import numpy as np

from src.inference.kv_cache import KVCache


def _layer_norm(x, gamma, beta, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return gamma * ((x - mean) / np.sqrt(var + eps)) + beta


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * (x * x * x))))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


ACTIVATIONS = {
    "gelu": _gelu,
    "relu": lambda x: np.maximum(0, x),
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
}


class _Layer:

    def __init__(self, w, prefix):
        g = lambda name: np.ascontiguousarray(w[f"{prefix}.{name}"], dtype=np.float32)

        self.ln1_g = g("norm1.gamma")
        self.ln1_b = g("norm1.beta")
        self.wqkv = np.concatenate([g("attn.Wq"), g("attn.Wk"), g("attn.Wv")], axis=1)
        self.wo = g("attn.Wo")
        self.ln2_g = g("norm2.gamma")
        self.ln2_b = g("norm2.beta")
        self.fc1_k = g("fc1.kernel")
        self.fc1_b = g("fc1.bias")
        self.fc2_k = g("fc2.kernel")
        self.fc2_b = g("fc2.bias")


class InferenceModel:

    def __init__(self, config, weights):
        if config.activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation '{config.activation}'")

        self.config = config
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.context_length = config.context_length
        self.eps = config.norm_eps
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))
        self.act = ACTIVATIONS[config.activation]

        f32 = lambda name: np.ascontiguousarray(weights[name], dtype=np.float32)

        self.tok_emb = f32("token_emb.embedding")
        self.rope = None
        self.pos_emb = None

        if getattr(config, "position_encoding", "learned") == "rope":
            self.rope = config.rotary_tables()

            if "pos_emb.embedding" in weights:
                raise ValueError("rope model must not carry learned positional embeddings")
        else:
            self.pos_emb = f32("pos_emb.embedding")
        self.ln_f_g = f32("final_norm.gamma")
        self.ln_f_b = f32("final_norm.beta")
        self.layers = [_Layer(weights, f"blocks.{i}") for i in range(config.n_layers)]

        if config.tie_weights:
            self.head_w = self.tok_emb.T
            self.head_b = None
        else:
            self.head_w = f32("lm_head.kernel")
            self.head_b = f32("lm_head.bias")

        if self.tok_emb.shape != (config.vocab_size, config.d_model):
            raise ValueError("token embedding shape does not match config")

        if self.pos_emb is not None and self.pos_emb.shape != (config.context_length, config.d_model):
            raise ValueError("positional embedding shape does not match config")

    @property
    def nbytes(self):
        total = self.tok_emb.nbytes + self.ln_f_g.nbytes + self.ln_f_b.nbytes

        if self.pos_emb is not None:
            total += self.pos_emb.nbytes

        if self.head_b is not None:
            total += self.head_w.nbytes + self.head_b.nbytes

        for layer in self.layers:
            total += sum(v.nbytes for v in vars(layer).values())

        return total

    def new_cache(self, capacity):
        return KVCache(self.n_layers, self.n_heads, self.head_dim, min(capacity, self.context_length))

    def _logits(self, x):
        out = _layer_norm(x, self.ln_f_g, self.ln_f_b, self.eps) @ self.head_w

        if self.head_b is not None:
            out = out + self.head_b

        return out

    def _ffn(self, layer, x):
        h = _layer_norm(x, layer.ln2_g, layer.ln2_b, self.eps)
        return self.act(h @ layer.fc1_k + layer.fc1_b) @ layer.fc2_k + layer.fc2_b

    def forward_chunk(self, token_ids, cache, want_logits=True):
        ids = np.asarray(token_ids, dtype=np.int64)
        T = ids.size
        p0 = cache.length

        if T == 0:
            raise ValueError("empty chunk")

        if p0 + T > cache.capacity:
            raise ValueError("chunk exceeds kv cache capacity")

        x = self.tok_emb[ids]
        L = p0 + T

        if self.pos_emb is not None:
            x = x + self.pos_emb[p0:L]
        else:
            cos, sin = self.rope.tables(np.arange(p0, L))

        mask = None
        if T > 1:
            mask = np.arange(L)[None, :] > (p0 + np.arange(T))[:, None]

        H, hd, d = self.n_heads, self.head_dim, self.d_model

        for li, layer in enumerate(self.layers):
            h = _layer_norm(x, layer.ln1_g, layer.ln1_b, self.eps)
            qkv = h @ layer.wqkv

            q = qkv[:, :d].reshape(T, H, hd).transpose(1, 0, 2)
            k = qkv[:, d:2 * d].reshape(T, H, hd).transpose(1, 0, 2)

            if self.rope is not None:
                q = self.rope.rotate(q, cos, sin)
                k = self.rope.rotate(k, cos, sin)

            cache.k[li, :, p0:L] = k
            cache.v[li, :, p0:L] = qkv[:, 2 * d:].reshape(T, H, hd).transpose(1, 0, 2)

            scores = (q @ cache.k[li, :, :L].transpose(0, 2, 1)) * self.scale

            if mask is not None:
                scores[:, mask] = -1e9

            ctx = _softmax(scores) @ cache.v[li, :, :L]
            x = x + ctx.transpose(1, 0, 2).reshape(T, d) @ layer.wo
            x = x + self._ffn(layer, x)

        cache.length = L

        if not want_logits:
            return None

        return self._logits(x[-1:])[0]

    def prefill(self, token_ids, cache, chunk_size=256):
        ids = np.asarray(token_ids, dtype=np.int64)
        logits = None

        for start in range(0, ids.size, chunk_size):
            end = min(start + chunk_size, ids.size)
            logits = self.forward_chunk(ids[start:end], cache, want_logits=(end == ids.size))

        return logits

    def decode_batch(self, token_ids, caches):
        B = len(caches)
        ids = np.asarray(token_ids, dtype=np.int64)
        pos = np.array([c.length for c in caches], dtype=np.int64)

        for c in caches:
            if c.length + 1 > c.capacity:
                raise ValueError("kv cache capacity exceeded")

        x = self.tok_emb[ids]

        if self.pos_emb is not None:
            x = x + self.pos_emb[pos]
        else:
            cos, sin = self.rope.tables(pos)
            q_cos, q_sin = cos[:, None, None, :], sin[:, None, None, :]
            k_cos, k_sin = cos[:, None, :], sin[:, None, :]

        H, hd, d = self.n_heads, self.head_dim, self.d_model

        for li, layer in enumerate(self.layers):
            h = _layer_norm(x, layer.ln1_g, layer.ln1_b, self.eps)
            qkv = h @ layer.wqkv

            q = qkv[:, :d].reshape(B, H, 1, hd)
            k = qkv[:, d:2 * d].reshape(B, H, hd)
            v = qkv[:, 2 * d:].reshape(B, H, hd)

            if self.rope is not None:
                q = self.rope.rotate(q, q_cos, q_sin)
                k = self.rope.rotate(k, k_cos, k_sin)

            ctx = np.empty((B, d), dtype=np.float32)

            for b, c in enumerate(caches):
                n = int(pos[b])
                c.k[li, :, n] = k[b]
                c.v[li, :, n] = v[b]

                scores = (q[b] @ c.k[li, :, :n + 1].transpose(0, 2, 1)) * self.scale
                ctx[b] = (_softmax(scores) @ c.v[li, :, :n + 1]).reshape(d)

            x = x + ctx @ layer.wo
            x = x + self._ffn(layer, x)

        for c in caches:
            c.length += 1

        return self._logits(x)
