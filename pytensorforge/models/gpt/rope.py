import threading

import numpy as _host_np

from pytensorforge.backend import get_device, xp
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
        self._on_device = {}

        inv_freq = self.theta ** (-_host_np.arange(0, self.half, dtype=_host_np.float64) / self.half)
        positions = _host_np.arange(self.max_positions, dtype=_host_np.float64)

        if scaling == "linear":
            positions = positions / self.factor

        angles = positions[:, None] * inv_freq[None, :]

        self.cos = _host_np.cos(angles).astype(_host_np.float32)
        self.sin = _host_np.sin(angles).astype(_host_np.float32)

    def _device_tables(self):
        key = get_device()
        cached = self._on_device.get(key)

        if cached is None:
            cached = (xp.asarray(self.cos), xp.asarray(self.sin))
            self._on_device[key] = cached

        return cached

    def tables(self, positions):
        """cos/sin rows for explicit positions (host array-like), as arrays on the active device."""
        positions = _host_np.asarray(positions, dtype=_host_np.int64)

        if positions.size and (positions.min() < 0 or positions.max() >= self.max_positions):
            raise ValueError(f"position outside the rotary table (0..{self.max_positions - 1})")

        cos, sin = self._device_tables()
        idx = xp.asarray(positions)

        return cos[idx], sin[idx]

    def slice(self, start, stop):
        """cos/sin rows for the contiguous positions start..stop-1 (no host-to-device copy)."""
        if start < 0 or stop > self.max_positions or stop < start:
            raise ValueError(f"position outside the rotary table (0..{self.max_positions - 1})")

        cos, sin = self._device_tables()

        return cos[start:stop], sin[start:stop]

    def rotate(self, x, cos, sin, inverse=False):
        h = self.half
        x1 = x[..., :h]
        x2 = x[..., h:]
        out = xp.empty_like(x)

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
