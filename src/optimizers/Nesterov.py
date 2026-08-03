import numpy as np


class Nesterov:

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, params):

        for p in params:

            if not p.requires_grad:
                continue

            if id(p) not in self.velocity:
                self.velocity[id(p)] = np.zeros_like(p.data)

            v_prev = self.velocity[id(p)].copy()

            self.velocity[id(p)] = (
                self.momentum * self.velocity[id(p)]
                - self.lr * p.grad
            )

            p.data += (
                -self.momentum * v_prev
                + (1 + self.momentum) * self.velocity[id(p)]
            )

    def zero_grad(self, params):
        for p in params:
            p.zero_grad()