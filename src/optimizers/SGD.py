from src.optimizer.Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, grads, learning_rate = 0.01):
        super().__init__(parameters, grads, learning_rate)

    def step(self):
        for p, grad in zip(self.parameters, self.grads):
            p.data -= self.learning_rate * grad.data

class SGDescent:
    def __call__(self, learning_rate = 0.01):
        pass