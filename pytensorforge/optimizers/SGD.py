from pytensorforge.optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, grads, learning_rate = 0.01):
        super().__init__(parameters, grads, learning_rate)

    def step(self):
        for p, grad in zip(self.parameters, self.grads):
            p.data -= self.learning_rate * grad.data

class SGDescent:
    def __init__(self, lr=0.01): self.lr = lr

    def step(self, params):
        for p in params:
            if p.requires_grad:
                p.data -= self.lr * p.grad

    def zero_grad(self, params):
        for p in params:
            p.zero_grad()

    def state_dict(self):
        return {
            "lr": self.lr
        }

    def load_state_dict(self, state):
        self.lr = float(state["lr"])
