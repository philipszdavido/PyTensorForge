import numpy as np


def flatten(state):
    out = {}

    for layer_name, params in state.items():

        for param_name, value in params.items():
            out[f"{layer_name}.{param_name}"] = value

    return out

class ModelIO:

    @staticmethod
    def save(model, path):

        np.savez_compressed(
            path,
            **flatten(model.state_dict())
        )

    @staticmethod
    def load(model, path):

        data = np.load(path)

        state = {}

        for key in data.files:

            layer, param = key.split(".")

            if layer not in state:
                state[layer] = {}

            state[layer][param] = data[key]

        model.load_state_dict(state)

        return model