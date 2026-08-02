from src.optimizer.Optimizer import Optimizer


class Batch(Optimizer):
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            layer.step()