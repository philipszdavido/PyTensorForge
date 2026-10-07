import numpy as np

class SGDMomentum:

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

            v = self.velocity[id(p)]

            v[:] = self.momentum * v - self.lr * p.grad

            p.data += v

    def zero_grad(self, params):
        for p in params:
            p.zero_grad()