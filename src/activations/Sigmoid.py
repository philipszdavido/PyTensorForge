from src.activations.Activation import Activation

class Sigmoid(Activation):
    def __call__(self, x):
        return x.sigmoid()