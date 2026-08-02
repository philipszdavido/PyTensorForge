from .mse import MSE
from .CategoricalCrossEntropy import CategoricalCrossEntropy
from .bce import BinaryCrossEntropy

losses = {
    "mse": MSE,
    "mean_squared_error": MSE,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "binary_crossentropy": BinaryCrossEntropy,
}
