import numpy as np

from src.activations.Activation import Activation

class Sigmoid(Activation):
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))