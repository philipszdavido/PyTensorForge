class Optimizer:
    def __init__(self, parameters, grads, learning_rate):
        self.parameters = parameters
        self.grads = grads
        self.learning_rate = learning_rate

    def zero_out(self):
        for p in self.grads:
            p.zeros(p.shape)

    def step(self):
        raise NotImplementedError

class SerialOptimizer:
    def state_dict(self):
        raise NotImplementedError
    def load_state_dict(self, state):
        raise NotImplementedError