import numpy as np

from src.initializers.Initializer import Initializer


# class GlorotUniform:
#     def __call__(self, shape):
#         fan_in, fan_out = shape
#         limit = np.sqrt(6 / (fan_in + fan_out))
#         return np.random.uniform(-limit, limit, shape)

# class GlorotUniform(Initializer):
#     def __call__(self, shape):
#         fan_in, fan_out = shape
#         limit = np.sqrt(6.0 / (fan_in + fan_out))
#         return np.random.uniform(
#             -limit,
#             limit,
#             shape,
#         ).astype(np.float32)

class GlorotUniform:
    def __call__(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).astype(np.float32)