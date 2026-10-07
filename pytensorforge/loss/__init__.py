from .CrossEntropyWithLogitsLoss import CrossEntropyWithLogitsLoss
from .MSE import MSE
from .CategoricalCrossEntropy import CategoricalCrossEntropy
from .SparseCategoricalCrossEntropy import SparseCategoricalCrossEntropy
from .bce import BinaryCrossEntropy
from .mae import MAE
from .CrossEntropyLoss import CrossEntropyLoss

losses = {
    "mse": MSE(),
    "mean_squared_error": MSE(),
    "mean_absolute_error": MAE(),
    "categorical_crossentropy": CategoricalCrossEntropy(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "cross_entropy": CrossEntropyLoss(),
    "sparse_categorical_crossentropy": SparseCategoricalCrossEntropy(),
    "cross_entropy_with_logits": CrossEntropyWithLogitsLoss()
}
