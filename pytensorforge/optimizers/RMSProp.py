import numpy as np


class RMSProp:

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {}

    def step(self, params):

        for p in params:

            if not p.requires_grad:
                continue

            if id(p) not in self.cache:
                self.cache[id(p)] = np.zeros_like(p.data)

            self.cache[id(p)] = (
                self.beta * self.cache[id(p)]
                + (1 - self.beta) * (p.grad ** 2)
            )

            p.data -= (
                self.lr
                * p.grad
                / (np.sqrt(self.cache[id(p)]) + self.eps)
            )

    def zero_grad(self, params):
        for p in params:
            p.zero_grad()