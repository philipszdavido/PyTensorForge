from .ELU import ELU
from .ReLU import ReLU
from .SELU import SELU
from .Sigmoid import Sigmoid
from .Tanh import Tanh

activation_fns = {
    "relu": ReLU(),
    "tanh": Tanh(),
    "sigmoid": Sigmoid(),
    "elu": ELU(),
    "selu": SELU(),
}
