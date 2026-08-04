from src.neural.Parameter import Parameter
from src.initializers import initializer_fns

class Layer:
    def __init__(self):
        self.built = False

        self.weights = []
        self.trainable_weights = []
        self.non_trainable_weights = []

    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True

        return self.call(inputs)

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError

    def add_weight(
        self,
        shape,
        initializer="glorot_uniform",
        trainable=True,
        name=None,
    ):

        value = initializer_fns[initializer](shape)

        parameter = Parameter(
            data=value,
            trainable=trainable,
            name=name,
        )

        self.weights.append(parameter)

        if trainable:
            self.trainable_weights.append(parameter)
        else:
            self.non_trainable_weights.append(parameter)

        return parameter

    def parameters(self):
        return self.trainable_weights

    def state_dict(self):
        state = {}

        for p in self.parameters():
            state[p.name] = p.data.copy()

        return state

    def load_state_dict(self, state):

        if not state:
            return

        for p in self.parameters():

            if p.name in state:
                p.data[:] = state[p.name]

    def compute_output_shape(self, input_shape):
        raise NotImplementedError