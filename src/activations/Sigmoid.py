from src.activations.Activation import Activation
from src.core.Tensor import Tensor
from src.math.sigmoid import sigmoid


class Sigmoid(Activation):
    @staticmethod
    def forward(x):
        sig = sigmoid(x.data)

        out = Tensor(
            sig,
            requires_grad=x.requires_grad,
            parents=(x,),
            op="Sigmoid",
        )

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * sig * (1 - sig)

        out._backward = _backward

        return out

    def __call__(self, x):
        return x.sigmoid()