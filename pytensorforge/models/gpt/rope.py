import threading

import numpy as np


ROPE_SCALING = ("none", "linear", "ntk")


def scaled_theta(theta, head_dim, scaling, factor):
    if scaling == "ntk":
        return float(theta) * float(factor) ** (head_dim / (head_dim - 2))

    return float(theta)


class RotaryTables:

    def __init__(self, head_dim, max_positions, theta=10000.0, scaling="none", factor=1.0):
        if head_dim % 2 != 0:
            raise ValueError("rotary position encoding needs an even head dimension")

        if scaling not in ROPE_SCALING:
            raise ValueError(f"unknown rope scaling '{scaling}'; choose one of {ROPE_SCALING}")

        if not factor >= 1.0:
            raise ValueError("rope scaling factor must be >= 1")

        self.head_dim = head_dim
        self.half = head_dim // 2
        self.scaling = scaling
        self.factor = float(factor)
        self.theta = scaled_theta(theta, head_dim, scaling, factor)
        self.max_positions = int(max_positions)

        inv_freq = self.theta ** (-np.arange(0, self.half, dtype=np.float64) / self.half)
        positions = np.arange(self.max_positions, dtype=np.float64)

        if scaling == "linear":
            positions = positions / self.factor

        angles = positions[:, None] * inv_freq[None, :]

        self.cos = np.cos(angles).astype(np.float32)
        self.sin = np.sin(angles).astype(np.float32)

    def tables(self, positions):
        positions = np.asarray(positions, dtype=np.int64)

        if positions.size and (positions.min() < 0 or positions.max() >= self.max_positions):
            raise ValueError(f"position outside the rotary table (0..{self.max_positions - 1})")

        return self.cos[positions], self.sin[positions]

    def rotate(self, x, cos, sin, inverse=False):
        h = self.half
        x1 = x[..., :h]
        x2 = x[..., h:]
        out = np.empty_like(x)

        if inverse:
            out[..., :h] = x1 * cos + x2 * sin
            out[..., h:] = x2 * cos - x1 * sin
        else:
            out[..., :h] = x1 * cos - x2 * sin
            out[..., h:] = x2 * cos + x1 * sin

        return out


_cache = {}
_cache_lock = threading.Lock()


def rotary_tables(head_dim, max_positions, theta=10000.0, scaling="none", factor=1.0):
    key = (int(head_dim), int(max_positions), float(theta), scaling, float(factor))

    with _cache_lock:
        tables = _cache.get(key)

        if tables is None:
            tables = RotaryTables(head_dim, max_positions, theta, scaling, factor)
            _cache[key] = tables

        return tables
