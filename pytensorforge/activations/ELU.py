from pytensorforge.backend import xp
from pytensorforge.activations.Activation import Activation


class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x):
        return xp.where(x > 0, x, self.alpha * (xp.exp(x) - 1))