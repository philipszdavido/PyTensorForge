import numpy as np

from src.core.Tensor import Tensor, grad_enabled, no_grad


def checkpoint(fn, x):
    if not grad_enabled():
        return fn(x)

    rng_state = np.random.get_state()

    with no_grad():
        y_data = fn(Tensor(x.data, requires_grad=False)).data

    out = Tensor(y_data, requires_grad=True, parents=(x,), op="Checkpoint")

    def _backward():
        resume_state = np.random.get_state()
        np.random.set_state(rng_state)

        try:
            x_local = Tensor(x.data, requires_grad=True)
            y = fn(x_local)
        finally:
            np.random.set_state(resume_state)

        y.backward(out.grad, release=True)

        if x.requires_grad:
            x.grad += x_local.grad

    out._backward = _backward

    return out

