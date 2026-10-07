from pytensorforge.activations.Activation import Activation

class GELU(Activation):

    def __call__(self, x):
        return x.gelu()