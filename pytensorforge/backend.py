"""Compute backend: NumPy on the CPU, CuPy on NVIDIA GPUs, MLX on Apple-silicon GPUs.

Every array operation that runs on the device goes through ``xp``, a thin proxy that forwards to
whichever array module is active. Everything that must stay on the host (data loading, tokenizers,
random-number generation, checkpoint files) keeps using plain NumPy.

    import pytensorforge as ptf
    ptf.set_device("cuda")        # NVIDIA, or "cuda:1"
    ptf.set_device("mps")         # Apple-silicon Mac (MLX)
    ptf.set_device("auto")        # CUDA if present, else the Mac GPU, else the CPU

Rules the rest of the framework follows:

* The device is chosen once, before any model is built. Tensors and parameters live on it.
* Weights are initialised with the host NumPy RNG and then copied to the device, so a given seed
  produces the same starting weights on the CPU and on the GPU.
* Checkpoints, exported weights and optimizer state are always host NumPy arrays, so a checkpoint
  written on a GPU loads on a CPU and the other way round.
"""
import os
import threading

import numpy as _np

DEVICES = ("cpu", "cuda", "mps", "auto")

_lock = threading.Lock()
_xp = _np
_cp = None
_cupyx = None
_mx = None
_name = "cpu"
_index = 0


class BackendError(RuntimeError):
    pass


def _parse(spec):
    spec = ("cpu" if spec is None else str(spec)).strip().lower()

    if spec in ("mps", "metal", "mlx", "apple"):
        return "mps", 0

    if spec in ("gpu", "cuda"):
        return "cuda", 0

    if spec.startswith("cuda:") or spec.startswith("gpu:"):
        tail = spec.split(":", 1)[1]

        if not tail.isdigit():
            raise ValueError(f"invalid device '{spec}'; use cpu, cuda, cuda:<index>, mps or auto")

        return "cuda", int(tail)

    if spec in ("cpu", "auto"):
        return spec, 0

    raise ValueError(f"unknown device '{spec}'; use cpu, cuda, cuda:<index>, mps or auto")


def _load_cupy(tf32):
    if tf32 is not None:
        os.environ["CUPY_TF32"] = "1" if tf32 else "0"

    try:
        import cupy
        import cupyx
    except Exception as exc:
        raise BackendError(
            "the cuda device needs CuPy. Install the build that matches your CUDA toolkit, for example "
            "`pip install cupy-cuda12x` (CUDA 12) or `pip install cupy-cuda11x` (CUDA 11); "
            f"import failed with: {exc}"
        ) from exc

    return cupy, cupyx


def _load_mlx():
    try:
        import mlx.core as mx
    except Exception as exc:
        raise BackendError(
            "the mps device needs MLX, which runs on Apple-silicon Macs (M1 or newer, native arm64 Python). "
            f"Install it with `pip install mlx`; import failed with: {exc}"
        ) from exc

    return mx


def _mlx_gpu_available(mx):
    try:
        return bool(mx.is_available(mx.gpu))
    except Exception:
        pass

    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def cuda_available():
    try:
        cupy, _ = _load_cupy(None)
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def mps_available():
    try:
        return _mlx_gpu_available(_load_mlx())
    except Exception:
        return False


# --------------------------------------------------------------------------------------------------------------
# MLX compatibility layer: presents MLX through the NumPy-style calls the framework uses.
# --------------------------------------------------------------------------------------------------------------

def _mx_dtype(mx, d):
    """numpy / python dtype -> MLX dtype. float64 maps to float32 because the Apple GPU has no float64."""
    if d is None or isinstance(d, mx.Dtype):
        return d

    if d is bool:
        return mx.bool_

    name = _np.dtype(d).name

    if name == "float64":
        return mx.float32

    return getattr(mx, name)


class _MLXNamespace:
    """The subset of the NumPy API the framework uses, on top of mlx.core."""

    def __init__(self, mx):
        self._mx = mx
        self.float64 = mx.float32          # no float64 on the GPU
        self.ndarray = mx.array
        self.errstate = _np.errstate

    # -- dtype aware constructors ---------------------------------------------------------------------------
    def dtype(self, d):
        return _mx_dtype(self._mx, d)

    def asarray(self, a, dtype=None):
        mx = self._mx
        dtype = _mx_dtype(mx, dtype)

        if isinstance(a, mx.array):
            return a if dtype is None or a.dtype == dtype else a.astype(dtype)

        if not isinstance(a, (_np.ndarray, int, float, bool)):
            a = _np.asarray(a)          # lists of numpy scalars, tuples, ... go through NumPy first

        if isinstance(a, _np.ndarray) and a.dtype == _np.float64 and dtype is None:
            dtype = mx.float32

        return mx.array(a, dtype=dtype) if dtype is not None else mx.array(a)

    array = asarray

    def ascontiguousarray(self, a, dtype=None):
        return self.asarray(a, dtype)

    def _shaped(self, name):
        fn = getattr(self._mx, name)

        def call(shape, dtype=None, **kw):
            return fn(shape, dtype=_mx_dtype(self._mx, dtype) or self._mx.float32, **kw)

        return call

    def zeros(self, shape, dtype=None):
        return self._mx.zeros(shape, dtype=_mx_dtype(self._mx, dtype) or self._mx.float32)

    def ones(self, shape, dtype=None):
        return self._mx.ones(shape, dtype=_mx_dtype(self._mx, dtype) or self._mx.float32)

    def empty(self, shape, dtype=None):
        return self.zeros(shape, dtype)

    def empty_like(self, a):
        return self._mx.zeros_like(a)

    def arange(self, *args, dtype=None):
        return self._mx.arange(*args, dtype=_mx_dtype(self._mx, dtype)) if dtype is not None else self._mx.arange(*args)

    def dot(self, a, b):
        return self._mx.matmul(a, b)

    def diagflat(self, a):
        return self._mx.diag(a.reshape(-1))

    def __getattr__(self, name):
        return getattr(self._mx, name)


def _make_mlx_namespace(mx):
    return _MLXNamespace(mx)


def set_device(spec=None, tf32=None, memory_limit_gb=None, cache_limit_mb=None):
    """Choose the compute device. Call it before building a model or creating tensors.

    ``spec``: ``cpu``, ``cuda``, ``cuda:<index>``, ``mps`` (Apple-silicon GPU through MLX) or ``auto`` (CUDA if present,
    else the Mac GPU, else the CPU). ``None`` reads the PTF_DEVICE environment variable and falls back to ``cpu``.
    ``tf32``: CUDA only. Allow TensorFloat-32 matrix multiplies on Ampere or newer GPUs (faster, slightly lower fp32
    precision). Takes effect only if CuPy has not been imported yet; ``None`` leaves it alone.
    """
    global _xp, _cp, _cupyx, _mx, _name, _index

    if spec is None:
        spec = os.environ.get("PTF_DEVICE", "cpu")

    kind, index = _parse(spec)

    with _lock:
        if kind == "auto":
            if cuda_available():
                kind, index = "cuda", 0
            elif mps_available():
                kind, index = "mps", 0
            else:
                kind, index = "cpu", 0

        if kind == "cpu":
            _xp, _cp, _cupyx, _mx, _name, _index = _np, None, None, None, "cpu", 0
            return get_device()

        if kind == "mps":
            mx = _load_mlx()

            if _mlx_gpu_available(mx):
                mx.set_default_device(mx.gpu)
            elif os.environ.get("PTF_MLX_ALLOW_CPU") != "1":
                raise BackendError(
                    "MLX is installed but no Apple GPU is available (it needs an Apple-silicon Mac running macOS "
                    "13.5 or newer, with a native arm64 Python)"
                )

            _xp, _cp, _cupyx, _mx, _name, _index = _make_mlx_namespace(mx), None, None, mx, "mps", 0
            _apply_mlx_limits(mx, memory_limit_gb, cache_limit_mb)
            return get_device()

        cupy, cupyx = _load_cupy(tf32)

        try:
            count = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:
            raise BackendError(f"no usable CUDA driver or GPU found: {exc}") from exc

        if count == 0:
            raise BackendError("CuPy is installed but no CUDA device is visible")

        if index >= count:
            raise BackendError(f"cuda:{index} requested but only {count} CUDA device(s) are visible")

        cupy.cuda.Device(index).use()
        _xp, _cp, _cupyx, _mx, _name, _index = cupy, cupy, cupyx, None, "cuda", index

    return get_device()


def _system_memory_bytes():
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _apply_mlx_limits(mx, memory_limit_gb, cache_limit_mb):
    total = _system_memory_bytes()

    if memory_limit_gb is not None:
        limit = int(float(memory_limit_gb) * (1 << 30))
    elif total:
        limit = total // 2
    else:
        limit = None

    cache = int(float(cache_limit_mb if cache_limit_mb is not None else 256) * (1 << 20))

    try:
        if limit:
            mx.set_memory_limit(limit)

        mx.set_cache_limit(cache)
    except Exception:
        pass


def bind_thread():
    """Worker threads that touch device arrays call this first (CuPy's current device and MLX's default device are
    per-thread)."""
    if _cp is not None:
        _cp.cuda.Device(_index).use()
    elif _mx is not None and _mlx_gpu_available(_mx):
        _mx.set_default_device(_mx.gpu)


def get_device():
    if _name == "cpu":
        return "cpu"

    if _name == "mps":
        return "mps"

    return f"cuda:{_index}"


def is_gpu():
    """True for any accelerator (CUDA or Apple GPU)."""
    return _name != "cpu"


def is_cuda():
    return _cp is not None


def is_mps():
    return _mx is not None


def same_device(spec):
    """True when ``spec`` names the device that is already active (``auto`` always matches)."""
    kind, index = _parse(spec)

    if kind == "auto":
        return True

    if kind in ("cpu", "mps"):
        return get_device() == kind

    return get_device() == f"cuda:{index}"


class _XP:
    """Forwards attribute access to the active array module (numpy, cupy or the MLX compatibility layer)."""

    # CuPy and MLX have no errstate and no need for one: GPU float errors never raise.
    errstate = staticmethod(_np.errstate)

    def __getattr__(self, name):
        return getattr(_xp, name)


xp = _XP()


def to_device(array, dtype=None):
    """Host or device array-like -> array on the active device."""
    return _xp.asarray(array, dtype=dtype)


def _is_device_array(array):
    if _cp is not None:
        return isinstance(array, _cp.ndarray)

    if _mx is not None:
        return isinstance(array, _mx.array)

    return False


def to_numpy(array):
    """Any array (host or device) -> host NumPy array. Device arrays are copied; host arrays are returned as is."""
    if _cp is not None and isinstance(array, _cp.ndarray):
        return _cp.asnumpy(array)

    if _mx is not None and isinstance(array, _mx.array):
        return _np.array(array)

    return _np.asarray(array)


def copy_to_numpy(array):
    """Like to_numpy but always returns a fresh host copy (used for checkpoints and exports)."""
    if _is_device_array(array):
        return _np.array(to_numpy(array), copy=True)

    return _np.array(array, copy=True)


def clone(array):
    """An independent copy on the active device (MLX arrays have no .copy())."""
    if _mx is not None:
        return _mx.array(array)

    return array.copy()


def zero_(array):
    """Set every element to 0 in place (MLX arrays have no .fill())."""
    array[...] = 0


def exp_(x):
    """exp(x), in place where the backend can (NumPy/CuPy). Always use the return value."""
    if _mx is not None:
        return _mx.exp(x)

    _xp.exp(x, out=x)

    return x


def assign(target, source):
    """target[...] = source, with source from the host or the device."""
    target[...] = _xp.asarray(source, dtype=target.dtype)


def scatter_add(target, indices, values):
    """np.add.at(target, indices, values) on every backend (duplicate indices accumulate), in place on target."""
    if _mx is not None:
        target[...] = target.at[indices].add(values)
        return

    if _cp is None:
        _np.add.at(target, indices, values)
        return

    if hasattr(_cp.add, "at"):
        _cp.add.at(target, indices, values)
    else:
        _cupyx.scatter_add(target, indices, values)


def scatter_add_rows(target, row_indices, rows):
    """target[row_indices[i]] += rows[i] for a 2-D target and a flat index vector (embedding gradients)."""
    if _mx is not None:
        target[...] = target.at[row_indices].add(rows)
    elif _cp is None:
        _np.add.at(target, row_indices, rows)
    else:
        _cupyx.scatter_add(target, row_indices, rows)


def materialize(*arrays):
    """Force pending computation. MLX is lazy: without this, queued work (and the activations it keeps alive) piles
    up until something reads a value. A no-op on NumPy and CuPy."""
    if _mx is None:
        return

    flat = []

    for a in arrays:
        if isinstance(a, (list, tuple)):
            flat.extend(a)
        elif a is not None:
            flat.append(a)

    if flat:
        _mx.eval(*flat)


def synchronize():
    """Block until queued device work finishes (no-op on the CPU). Needed for honest timings."""
    if _cp is not None:
        _cp.cuda.Stream.null.synchronize()
    elif _mx is not None:
        _mx.synchronize()


def empty_cache():
    if _cp is not None:
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    elif _mx is not None:
        try:
            _mx.clear_cache()
        except Exception:
            pass


def memory_stats():
    """Bytes currently used / reserved on the accelerator, or None on the CPU."""
    if _cp is not None:
        pool = _cp.get_default_memory_pool()

        return {"used_bytes": int(pool.used_bytes()), "reserved_bytes": int(pool.total_bytes())}

    if _mx is not None:
        try:
            return {"used_bytes": int(_mx.get_active_memory()), "reserved_bytes": int(_mx.get_cache_memory())}
        except Exception:
            return None

    return None


def device_info():
    if _mx is not None:
        info = {}

        try:
            info = dict(_mx.device_info())
        except Exception:
            try:
                info = dict(_mx.metal.device_info())
            except Exception:
                pass

        return {
            "device": "mps",
            "backend": "mlx",
            "name": str(info.get("device_name", "Apple GPU")),
            "total_memory_bytes": int(info["memory_size"]) if "memory_size" in info else None,
        }

    if _cp is None:
        return {"device": "cpu", "backend": "numpy", "name": "cpu", "total_memory_bytes": None}

    props = _cp.cuda.runtime.getDeviceProperties(_index)
    name = props["name"]

    if isinstance(name, bytes):
        name = name.decode()

    return {
        "device": get_device(),
        "backend": "cupy",
        "name": name,
        "total_memory_bytes": int(props["totalGlobalMem"]),
        "compute_capability": f"{props['major']}.{props['minor']}",
    }


def host_rng_sample_mask(shape, keep):
    """Dropout mask drawn with the host NumPy RNG (so seeding and checkpoint recompute stay exact), then copied over."""
    return to_device(_np.random.binomial(1, keep, size=shape).astype(_np.float32))
