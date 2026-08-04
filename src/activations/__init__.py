from .ELU import ELU
from .GELU import GELU
from .ReLU import ReLU
from .SELU import SELU
from .Sigmoid import Sigmoid
from .Tanh import Tanh
from .Softmax import Softmax

activation_fns = {
    "relu": ReLU(),
    "tanh": Tanh(),
    "sigmoid": Sigmoid(),
    "elu": ELU(),
    "selu": SELU(),
    "softmax": Softmax(),
    "gelu": GELU(),
}
