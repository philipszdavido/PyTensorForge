from pytensorforge.neural.Layer import Layer


class LastToken(Layer):

    def build(self, input_shape):
        pass

    def call(self, x):
        return x[:, -1, :]