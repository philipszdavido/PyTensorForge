import math

import numpy as _host_np

from pytensorforge.backend import is_cuda, is_mps, xp
from pytensorforge.core.Tensor import set_matmul_policy

_u32 = _host_np.uint32

PRECISIONS = ("fp32", "bf16", "fp16")
LOSS_SCALING = ("auto", "dynamic", "none")


def round_bf16(x):
    if is_mps():
        return xp.asarray(x, dtype=xp.float32).astype(xp.bfloat16).astype(xp.float32)   # native round-to-nearest-even

    x = xp.ascontiguousarray(xp.asarray(x, dtype=xp.float32))
    bits = x.view(_u32)
    finite = (bits & _u32(0x7F800000)) != _u32(0x7F800000)
    lsb = (bits >> _u32(16)) & _u32(1)
    rounded = (bits + _u32(0x7FFF) + lsb) & _u32(0xFFFF0000)
    is_nan = ~finite & ((bits & _u32(0x007FFFFF)) != _u32(0))
    special = (bits & _u32(0xFFFF0000)) | (is_nan.astype(_u32) * _u32(0x00400000))
    return xp.where(finite, rounded, special).astype(_u32).view(xp.float32)


def round_fp16(x):
    with xp.errstate(over="ignore"):
        return xp.asarray(x, dtype=xp.float32).astype(xp.float16).astype(xp.float32)


ROUNDERS = {"bf16": round_bf16, "fp16": round_fp16}


class MatmulPolicy:
    """Rounds matmul operands/results to a reduced precision. ``native_dtype`` is set when the device can
    run the product in that dtype directly (fp16 on a GPU), which is what makes mixed precision fast."""

    def __init__(self, precision, rounder, native_dtype=None):
        self.precision = precision
        self.rounder = rounder
        self.native_dtype = native_dtype

    def __call__(self, x):
        return self.rounder(x)


def make_policy(precision):
    if precision == "fp32":
        return None

    # fp16 products run natively on CUDA and on the Apple GPU; bf16 runs natively only through MLX
    native = None

    if precision == "fp16" and (is_cuda() or is_mps()):
        native = xp.float16
    elif precision == "bf16" and is_mps():
        native = xp.bfloat16

    return MatmulPolicy(precision, ROUNDERS[precision], native)


class autocast:

    def __init__(self, precision):
        if precision not in PRECISIONS:
            raise ValueError(f"unknown precision '{precision}'; choose one of {PRECISIONS}")

        self.precision = precision
        self._previous = None

    def __enter__(self):
        self._previous = set_matmul_policy(make_policy(self.precision))
        return self

    def __exit__(self, *exc):
        set_matmul_policy(self._previous)
        return False


class GradScaler:

    def __init__(
        self,
        enabled=True,
        init_scale=65536.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        min_scale=1.0,
        max_scale=2.0 ** 24,
    ):
        if init_scale <= 0 or growth_factor <= 1 or not 0 < backoff_factor < 1 or growth_interval < 1:
            raise ValueError("invalid loss scaler settings")

        self.enabled = bool(enabled)
        self.scale = float(init_scale) if enabled else 1.0
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.good_steps = 0
        self.skipped_steps = 0

    def loss_multiplier(self):
        return self.scale if self.enabled else 1.0

    def grads_finite(self, params):
        flags = [
            xp.isfinite(p._grad).all()
            for p in params
            if p.requires_grad and p._grad is not None
        ]

        if not flags:
            return True

        return bool(xp.stack(flags).all())

    def unscale_and_check(self, params):
        if not self.enabled:
            return True

        if not self.grads_finite(params):
            return False

        inv = 1.0 / self.scale

        for p in params:
            if p.requires_grad and p._grad is not None:
                p._grad *= inv

        return True

    def update(self, found_overflow):
        if not self.enabled:
            return

        if found_overflow:
            self.skipped_steps += 1
            self.good_steps = 0
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            return

        self.good_steps += 1

        if self.good_steps >= self.growth_interval:
            self.good_steps = 0
            self.scale = min(self.max_scale, self.scale * self.growth_factor)

    def state_dict(self):
        return {
            "enabled": self.enabled,
            "scale": self.scale,
            "good_steps": self.good_steps,
            "skipped_steps": self.skipped_steps,
        }

    def load_state_dict(self, state):
        if bool(state.get("enabled", False)) != self.enabled:
            raise ValueError("checkpoint loss-scaling mode does not match this run")

        self.scale = float(state["scale"])
        self.good_steps = int(state["good_steps"])
        self.skipped_steps = int(state["skipped_steps"])

        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("checkpoint carries an invalid loss scale")


def resolve_loss_scaling(precision, loss_scaling):
    if loss_scaling not in LOSS_SCALING:
        raise ValueError(f"unknown loss_scaling '{loss_scaling}'; choose one of {LOSS_SCALING}")

    if loss_scaling == "auto":
        return precision == "fp16"

    return loss_scaling == "dynamic"
