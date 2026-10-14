"""A strict stand-in for CuPy, so the GPU code paths can be exercised on a machine with no GPU.

It is NumPy underneath, but "device" arrays are a distinct subclass (DevArray) and the fake ``cupy`` module
refuses the same mistakes the real one refuses:

* handing a host ndarray to a device function (real CuPy raises TypeError; ``asarray``/``array`` is the way over),
* calling a NumPy function that is dispatched through ``__array_function__`` on a device array
  (np.add.at, np.savez, np.where ... on device data),
* implicit conversion with ``__array__`` of a non-ndarray device object is not modelled, but ``to_numpy`` is
  the only sanctioned way back to the host in the framework and the tests check for DevArray leaking into files.

It proves the framework keeps host and device data apart. It does not prove CUDA kernels compile or run, and
it says nothing about speed or memory.

    import test.fakegpu as fakegpu
    fakegpu.install()                 # puts fake `cupy` and `cupyx` into sys.modules
    pytensorforge.set_device("cuda")
"""
import sys
import threading
import types

import numpy as np

_local = threading.local()


class FakeGPUError(TypeError):
    pass


def _inside():
    return getattr(_local, "depth", 0) > 0


class _enter:
    def __enter__(self):
        _local.depth = getattr(_local, "depth", 0) + 1

    def __exit__(self, *exc):
        _local.depth -= 1
        return False


def _from_test_code():
    """Test scripts poke at p.data directly; only framework code is held to the strict rules."""
    f = sys._getframe(1)

    while f is not None:
        name = f.f_code.co_filename.replace("\\", "/")

        if not name.endswith("fakegpu.py") and "/numpy/" not in name:
            return "test/gpt/" in name or "test/word" in name or name.endswith("run_on_fakegpu.py")

        f = f.f_back

    return False


def _is_host(x):
    return isinstance(x, np.ndarray) and not isinstance(x, DevArray) and x.ndim > 0


def _check_inputs(name, args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, (list, tuple)):
            _check_inputs(name, a, {})
        elif _is_host(a):
            raise FakeGPUError(f"{name}: host ndarray passed to a device function (use asarray to move it over)")


class DevArray(np.ndarray):
    """A 'device' array."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        lenient = _from_test_code()

        for x in inputs:
            if _is_host(x) and not lenient:
                raise FakeGPUError(f"ufunc {ufunc.__name__}: host ndarray mixed with a device array")

        out = kwargs.get("out")

        if out is not None:
            for o in out:
                if _is_host(o):
                    raise FakeGPUError(f"ufunc {ufunc.__name__}: host array used as out= for a device op")

        with _enter():
            plain = [np.asarray(x) if isinstance(x, DevArray) else x for x in inputs]

            if out is not None:
                kwargs["out"] = tuple(np.asarray(o) if isinstance(o, DevArray) else o for o in out)

            result = getattr(ufunc, method)(*plain, **kwargs)

        return _wrap(result)

    def __array_function__(self, func, types_, args, kwargs):
        if not _inside() and not _from_test_code():
            raise FakeGPUError(f"numpy.{func.__name__} called on a device array from host code")

        for t in args:
            pass

        with _enter():
            return super().__array_function__(func, types_, args, kwargs)

    def get(self):
        return np.array(np.asarray(self))

    def __repr__(self):
        return "DevArray(" + np.asarray(self).__repr__() + ")"

    def __getitem__(self, idx):
        for i in (idx if isinstance(idx, tuple) else (idx,)):
            if _is_host(i) and i.dtype != bool and i.size > 1:
                # real CuPy accepts numpy index arrays too, but relying on it is a portability trap
                raise FakeGPUError("indexing a device array with a host ndarray")

        return super().__getitem__(idx)

    def __setitem__(self, idx, value):
        if _is_host(value) and not isinstance(value, DevArray):
            # CuPy does accept a numpy array on assignment (it copies it over)
            value = np.asarray(value)

        super().__setitem__(idx, value)


def _wrap(x):
    if isinstance(x, DevArray):
        return x

    if isinstance(x, np.ndarray):
        return x.view(DevArray)

    if isinstance(x, tuple):
        return tuple(_wrap(i) for i in x)

    if isinstance(x, list):
        return [_wrap(i) for i in x]

    return x


def _unwrap(x):
    if isinstance(x, DevArray):
        return np.asarray(x)

    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(i) for i in x)

    return x


class _Ufunc:
    """cupy.add etc: callable plus an 'at' attribute."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        _check_inputs(self._fn.__name__, args, kwargs)

        with _enter():
            return _wrap(self._fn(*[_unwrap(a) for a in args], **{k: _unwrap(v) for k, v in kwargs.items()}))

    def at(self, target, idx, values):
        with _enter():
            self._fn.at(_unwrap(target), _unwrap(idx), _unwrap(values))


class _Random:
    def __getattr__(self, name):
        raise FakeGPUError(f"cupy.random.{name} is not modelled: the framework must draw random numbers on the host")


class FakeCupy(types.ModuleType):
    """The module object that stands in for `cupy`."""

    def __init__(self):
        super().__init__("cupy")
        self.ndarray = DevArray
        self.random = _Random()
        self.__version__ = "0-fake"

        cuda = types.SimpleNamespace()
        cuda.Device = lambda i=0: types.SimpleNamespace(use=lambda: None)
        cuda.runtime = types.SimpleNamespace(
            getDeviceCount=lambda: 1,
            getDeviceProperties=lambda i: {"name": b"FakeGPU", "totalGlobalMem": 8 << 30, "major": 8, "minor": 6},
        )
        cuda.Stream = types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None))
        self.cuda = cuda

    def asnumpy(self, a):
        return a.get() if isinstance(a, DevArray) else np.asarray(a)

    def get_default_memory_pool(self):
        return types.SimpleNamespace(used_bytes=lambda: 0, total_bytes=lambda: 0, free_all_blocks=lambda: None)

    def get_default_pinned_memory_pool(self):
        return types.SimpleNamespace(free_all_blocks=lambda: None)

    def asarray(self, a, dtype=None):
        if isinstance(a, DevArray):
            return a if dtype is None or a.dtype == np.dtype(dtype) else a.astype(dtype)

        with _enter():
            return _wrap(np.array(a, dtype=dtype, copy=True))

    array = asarray

    def ascontiguousarray(self, a, dtype=None):
        if not isinstance(a, DevArray):
            raise FakeGPUError("ascontiguousarray: host ndarray passed to a device function")

        with _enter():
            return _wrap(np.ascontiguousarray(np.asarray(a), dtype=dtype))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        attr = getattr(np, name)

        if isinstance(attr, np.ufunc):
            return _Ufunc(attr)

        if isinstance(attr, type) or not callable(attr):
            return attr

        def call(*args, **kwargs):
            _check_inputs(name, args, kwargs)

            with _enter():
                return _wrap(attr(*[_unwrap(a) for a in args], **{k: _unwrap(v) for k, v in kwargs.items()}))

        call.__name__ = name

        return call


def _scatter_add(target, idx, values):
    with _enter():
        np.add.at(np.asarray(target), _unwrap(idx), _unwrap(values))


def install():
    cupy = FakeCupy()
    cupyx = types.ModuleType("cupyx")
    cupyx.scatter_add = _scatter_add
    sys.modules["cupy"] = cupy
    sys.modules["cupyx"] = cupyx

    return cupy
