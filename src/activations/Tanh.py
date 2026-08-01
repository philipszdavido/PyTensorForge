import numpy as np

from src.activations.Activation import Activation


class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)