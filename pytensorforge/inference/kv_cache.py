import numpy as np


class CacheFull(Exception):
    pass


class KVCache:

    def __init__(self, n_layers, n_heads, head_dim, capacity, dtype=np.float32):
        self.capacity = capacity
        self.k = np.empty((n_layers, n_heads, capacity, head_dim), dtype=dtype)
        self.v = np.empty((n_layers, n_heads, capacity, head_dim), dtype=dtype)
        self.length = 0

    @property
    def nbytes(self):
        return self.k.nbytes + self.v.nbytes

    def reset(self):
        self.length = 0


class KVCacheManager:

    def __init__(self, n_layers, n_heads, head_dim, max_bytes=None, dtype=np.float32):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_bytes = max_bytes
        self.dtype = np.dtype(dtype)
        self._caches = {}
        self.used_bytes = 0

    def bytes_for(self, capacity):
        return 2 * self.n_layers * self.n_heads * capacity * self.head_dim * self.dtype.itemsize

    def fits_at_all(self, capacity):
        return self.max_bytes is None or self.bytes_for(capacity) <= self.max_bytes

    def can_allocate(self, capacity):
        return self.max_bytes is None or self.used_bytes + self.bytes_for(capacity) <= self.max_bytes

    def allocate(self, key, capacity):
        if key in self._caches:
            raise KeyError(f"cache already allocated for {key}")

        if not self.can_allocate(capacity):
            raise CacheFull(f"kv cache budget of {self.max_bytes} bytes exhausted")

        cache = KVCache(self.n_layers, self.n_heads, self.head_dim, capacity, self.dtype)
        self._caches[key] = cache
        self.used_bytes += cache.nbytes

        return cache

    def release(self, key):
        cache = self._caches.pop(key, None)

        if cache is not None:
            self.used_bytes -= cache.nbytes

    @property
    def active(self):
        return len(self._caches)
