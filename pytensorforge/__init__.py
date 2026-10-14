try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("pytensorforge")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


from pytensorforge.backend import (  # noqa: E402
    BackendError,
    cuda_available,
    device_info,
    get_device,
    set_device,
    to_numpy,
)
