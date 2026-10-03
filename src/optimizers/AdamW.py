import numpy as np

from src.optimizers.Optimizer import SerialOptimizer


class AdamW(SerialOptimizer):

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {}
        self.v = {}

        self.t = 0

    def step(self, params):

        self.t += 1

        for pid, p in enumerate(params):

            if not p.requires_grad:
                continue

            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data)
                self.v[pid] = np.zeros_like(p.data)

            p.data *= (1 - self.lr * self.weight_decay)

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
            "t": self.t,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": {k: v.copy() for k, v in self.m.items()},
            "v": {k: v.copy() for k, v in self.v.items()},
        }

    def load_state_dict(self, state):
        self.t = state["t"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]
        self.weight_decay = state["weight_decay"]
        self.m = {k: v.copy() for k, v in state["m"].items()}
        self.v = {k: v.copy() for k, v in state["v"].items()}