import numpy as np

from src.activations.Activation import Activation


class ReLU(Activation):
    def __call__(self, x):
        return x.relu()