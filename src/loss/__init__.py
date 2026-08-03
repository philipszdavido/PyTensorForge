from .MSE import MSE
from .CategoricalCrossEntropy import CategoricalCrossEntropy
from .bce import BinaryCrossEntropy
from .mae import MAE

losses = {
    "mse": MSE(),
    "mean_squared_error": MSE(),
    "mean_absolute_error": MAE(),
    "categorical_crossentropy": CategoricalCrossEntropy(),
    "binary_crossentropy": BinaryCrossEntropy(),
}
