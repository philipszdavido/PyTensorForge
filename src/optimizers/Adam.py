import numpy as np

from src.optimizers.Optimizer import SerialOptimizer


class Adam(SerialOptimizer):

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}

        self.t = 0

    def step(self, params):

        self.t += 1

        for p in params:

            if not p.requires_grad:
                continue

            pid = id(p)

            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data)
                self.v[pid] = np.zeros_like(p.data)

            self.m[pid] = (
                self.beta1 * self.m[pid]
                + (1 - self.beta1) * p.grad
            )

            self.v[pid] = (
                self.beta2 * self.v[pid]
                + (1 - self.beta2) * (p.grad ** 2)
            )

            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            p.data -= (
                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )

    def zero_grad(self, params):
        for p in params:
            p.zero_grad()

    def state_dict(self):

        return {
            "m": self.m,
            "v": self.v,
            "t": self.t,
            "lr": self.lr
        }

    def load_state_dict(self, state):

        self.m = state["m"]
        self.v = state["v"]
        self.t = state["t"]
        self.lr = state["lr"]