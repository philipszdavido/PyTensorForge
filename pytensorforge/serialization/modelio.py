# import numpy as np
#
#
# def flatten(state):
#     out = {}
#
#     for layer_name, params in state.items():
#
#         for param_name, value in params.items():
#             out[f"{layer_name}.{param_name}"] = value
#
#     return out
#
# class ModelIO:
#
#     @staticmethod
#     def save(model, path, metadata = None):
#
#         np.savez_compressed(
#             path,
#             **flatten(model.state_dict())
#         )
#
#     @staticmethod
#     def save_data(data, path):
#
#         np.savez_compressed(
#             path,
#             **flatten(data)
#         )
#
#     @staticmethod
#     def load(model, path):
#
#         data = np.load(path)
#
#         state = {}
#
#         for key in data.files:
#
#             layer, param = key.split(".")
#
#             if layer not in state:
#                 state[layer] = {}
#
#             state[layer][param] = data[key]
#
#         model.load_state_dict(state)
#
#         return model

import numpy as np


def flatten(d, prefix=""):
    out = {}

    for k, v in d.items():

        key = f"{prefix}.{k}" if prefix else k

        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v

    return out


def unflatten(flat):
    out = {}

    for key, value in flat.items():

        parts = key.split(".")

        d = out

        for p in parts[:-1]:
            d = d.setdefault(p, {})

        d[parts[-1]] = value

    return out


class ModelIO:

    @staticmethod
    def save(model, path, metadata=None):

        state = {
            "model": model.state_dict()
        }

        if metadata is not None:
            state["metadata"] = metadata

        np.savez_compressed(
            path,
            **flatten(state)
        )

    @staticmethod
    def load(model, path):

        data = np.load(path, allow_pickle=True)

        flat = {
            k: data[k]
            for k in data.files
        }

        state = unflatten(flat)

        model.load_state_dict(state["model"])

        return state.get("metadata", None)

    @staticmethod
    def read(path):

        data = np.load(path, allow_pickle=True)

        flat = {
            k: data[k]
            for k in data.files
        }

        state = unflatten(flat)

        return state