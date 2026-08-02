import numpy as np

# x = Tensor([[1, 2],
#             [3, 4]], requires_grad=True)
#
# print(x.shape)

def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad

class Tensor:

    def __init__(
            self,
            data,
            requires_grad=True,
            parents=(),
            op=None,
    ):
        self.data = np.asarray(data, dtype=np.float32)

        self.grad = np.zeros_like(self.data)

        self.requires_grad = requires_grad

        self.parents = parents

        self.op = op

        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad.fill(0)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
        )

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="Add",
        )

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)

            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)

        out._backward = backward

        return out

    def __sub__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="Sub",
        )

        def _backward_():
            # out = self - other
            # 𝜹out/𝜹self = 1 - 0 = 1
            self.grad += out.grad * 1
            # 𝜹out/𝜹other = 0 - 1 = -1
            other.grad += out.grad * (-1)

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)

            if other.requires_grad:
                other.grad += unbroadcast(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="Mul",
        )

        def _backward_():
            # out = self * other
            # 𝜹out/𝜹self = other
            self.grad += out.grad * other.data
            # out = self * other
            # 𝜹out/𝜹other = self
            other.grad += out.grad * self.data

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(
                    out.grad * other.data,
                    self.shape,
                )

            if other.requires_grad:
                other.grad += unbroadcast(
                    out.grad * self.data,
                    other.shape,
                )

        out._backward = _backward
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, parents=(self, other), op="MatMul")
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __div__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="Div",
        )

        def _backward():
            # out = self / other
            # 𝜹out/𝜹self = 1/other
            # 𝜹out/𝜹other = self * (-1) * other ** (-2)
            self.grad += out.grad * ( 1 / other.data)
            other.grad += out.grad * (-self.data / (other.data ** 2))
        out._backward = _backward
        return out

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data ** other,
            requires_grad=self.requires_grad,
            parents=(self, ),
            op="Pow",
        )

        def _backward():
            # out = self ** other
            # 𝜹out/𝜹self = other * self ** (other - 1)
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward
        return out

    def sin(self):
        out = Tensor(
            np.sin(self.data),
            requires_grad=self.requires_grad,
            parents=(self, ),
            op="Sin",
        )

        def _backward():
            # out = sin(self)
            # 𝜹out/𝜹self = cos(self)
            self.grad += out.grad * np.cos(self.data)
        out._backward = _backward
        return out

    def mean(self):
        # data is an array
        out = Tensor(
            self.data.mean(),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Mean",
        )

        def _backward():
            self.grad += out.grad * np.ones_like(self.data) / self.data.size
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(
            self.data.sum(),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Sum",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self.data)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="Div",
        )

        def _backward_():
            if self.requires_grad:
                self.grad += out.grad / other.data

            if other.requires_grad:
                other.grad -= out.grad * self.data / (other.data ** 2)

        def _backward():

            if self.requires_grad:
                self.grad += unbroadcast(
                    out.grad / other.data,
                    self.shape,
                )

            if other.requires_grad:
                other.grad += unbroadcast(
                    -out.grad * self.data / (other.data ** 2),
                    other.shape,
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) - self

    def backward(self):
        # impl DAG topo
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()