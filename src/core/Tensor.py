import threading

import numpy as np

_state = threading.local()


def _noop():
    return None


def grad_enabled():
    return getattr(_state, "grad_enabled", True)


def matmul_policy():
    return getattr(_state, "matmul_policy", None)


def set_matmul_policy(fn):
    previous = matmul_policy()
    _state.matmul_policy = fn
    return previous


class no_grad:

    def __enter__(self):
        self._previous = grad_enabled()
        _state.grad_enabled = False
        return self

    def __exit__(self, *exc):
        _state.grad_enabled = self._previous
        return False


def _round_operand(x):
    policy = matmul_policy()
    return x if policy is None else policy(x)

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

def _topological_order(root):
    order = []
    visited = {id(root)}
    stack = [(root, iter(root.parents))]

    while stack:
        node, children = stack[-1]
        advanced = False

        for child in children:
            if id(child) not in visited:
                visited.add(id(child))
                stack.append((child, iter(child.parents)))
                advanced = True
                break

        if not advanced:
            stack.pop()
            order.append(node)

    return order


class Tensor:

    def __init__(
            self,
            data,
            requires_grad=True,
            parents=(),
            op=None,
    ):
        self.data = np.asarray(data, dtype=np.float32)

        self._grad = None

        if not grad_enabled():
            requires_grad = False
            parents = ()

        self.requires_grad = requires_grad

        self.parents = parents

        self.op = op

        self._bw = _noop

    @property
    def grad(self):
        if self._grad is None:
            self._grad = np.zeros_like(self.data)
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def _backward(self):
        return self._bw

    @_backward.setter
    def _backward(self, fn):
        self._bw = fn if self.requires_grad else _noop

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        if self._grad is not None:
            self._grad.fill(0)

    def __len__(self):
        return self.data.shape[0]

    # def __getitem__(self, idx):
    #     return Tensor(
    #         self.data[idx],
    #         requires_grad=self.requires_grad,
    #     )

    def __getitem__(self, idx):

        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)

        elif isinstance(idx, tuple):
            idx = tuple(
                i.data.astype(np.int64) if isinstance(i, Tensor) else i
                for i in idx
            )

        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Slice",
        )

        def _backward():
            if not self.requires_grad:
                return

            np.add.at(
                self.grad,
                idx,
                out.grad,
            )

        out._backward = _backward

        return out

    @staticmethod
    def zeros(shape, requires_grad=False):
        return Tensor(
            np.zeros(shape, dtype=np.float32),
            requires_grad=requires_grad,
        )

    @staticmethod
    def stack(tensors, axis=0):
        from src.ops.stack import Stack
        return Stack.forward(tensors, axis)

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

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

    # def __neg__(self):
    #     return self * -1

    def __neg__(self):
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Neg",
        )

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

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

    # def __matmul__(self, other):
    #     if not isinstance(other, Tensor):
    #         other = Tensor(other)
    #
    #     out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, parents=(self, other), op="MatMul")
    #     def _backward():
    #         self.grad += out.grad @ other.data.T
    #         other.grad += self.data.T @ out.grad
    #     out._backward = _backward
    #     return out

    def __matmul__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other)

        policy = matmul_policy()
        a = self.data if policy is None else policy(self.data)
        b = other.data if policy is None else policy(other.data)
        result = a @ b

        out = Tensor(
            result if policy is None else policy(result),
            requires_grad=self.requires_grad or other.requires_grad,
            parents=(self, other),
            op="MatMul",
        )

        def _backward():
            g = out.grad if policy is None else policy(out.grad)

            if self.requires_grad:
                grad = np.matmul(
                    g,
                    np.swapaxes(b, -1, -2),
                )

                if policy is not None:
                    grad = policy(grad)

                self.grad += Tensor.unbroadcast(
                    grad,
                    self.shape,
                )

            if other.requires_grad:
                grad = np.matmul(
                    np.swapaxes(a, -1, -2),
                    g,
                )

                if policy is not None:
                    grad = policy(grad)

                other.grad += Tensor.unbroadcast(
                    grad,
                    other.shape,
                )

        out._backward = _backward

        return out

    @staticmethod
    def unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, size in enumerate(shape):
            if size == 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad

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

    # def __pow__(self, other):
    #     if not isinstance(other, Tensor):
    #         other = Tensor(other)
    #
    #     out = Tensor(
    #         self.data ** other,
    #         requires_grad=self.requires_grad,
    #         parents=(self, ),
    #         op="Pow",
    #     )
    #
    #     def _backward():
    #         # out = self ** other
    #         # 𝜹out/𝜹self = other * self ** (other - 1)
    #         self.grad += out.grad * (other * self.data ** (other - 1))
    #     out._backward = _backward
    #     return out

    def __pow__(self, other):

        if isinstance(other, Tensor):
            other = other.data

        out = Tensor(
            self.data ** other,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Pow",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward

        return out

    def __rpow__(self, other):

        out = Tensor(
            other ** self.data,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="RPow",
        )

        def _backward():
            if self.requires_grad:
                self.grad += (
                        out.grad
                        * np.log(other)
                        * (other ** self.data)
                )

        out._backward = _backward

        return out

    def sqrt(self):

        out = Tensor(
            np.sqrt(self.data),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Sqrt",
        )

        def _backward():
            if not self.requires_grad:
                return

            self.grad += out.grad * (0.5 / np.sqrt(self.data))

        out._backward = _backward

        return out

    def masked_fill(self, mask, value):

        if isinstance(mask, Tensor):
            mask = mask.data

        mask = np.broadcast_to(mask, self.data.shape)

        out_data = self.data.copy()

        out_data[mask] = value

        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            parents=(self,),
            op="MaskedFill",
        )

        def _backward():

            if not self.requires_grad:
                return

            grad = out.grad.copy()

            grad[mask] = 0

            self.grad += grad

        out._backward = _backward

        return out

    # def __hash__(self):
    #     return id(self)

    __hash__ = object.__hash__

    def __eq__(self, other):

        if isinstance(other, Tensor):
            other = other.data

        return self.data == other

    def gelu(self):

        x = self.data

        c = 0.7978845608028654

        x2 = x * x

        tanh_inner = np.tanh(c * x * (1.0 + 0.044715 * x2))

        out = Tensor(
            0.5 * x * (1.0 + tanh_inner),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="GELU",
        )

        def _backward():
            if not self.requires_grad:
                return

            sech2 = 1.0 - tanh_inner * tanh_inner

            grad = 0.5 * (1.0 + tanh_inner) + (0.5 * c) * x * sech2 * (1.0 + (3.0 * 0.044715) * x2)

            self.grad += out.grad * grad

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

    # def mean(self):
    #     out = Tensor(
    #         self.data.mean(),
    #         requires_grad=self.requires_grad,
    #         parents=(self,),
    #         op="Mean",
    #     )
    #
    #     def _backward():
    #         self.grad += out.grad * np.ones_like(self.data) / self.data.size
    #     out._backward = _backward
    #     return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Mean",
        )

        def _backward():
            if not self.requires_grad:
                return

            grad = out.grad

            if axis is None:
                count = self.data.size
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                count = 1
                for ax in axes:
                    count *= self.data.shape[ax]

                if not keepdims:
                    for ax in sorted([a if a >= 0 else a + self.data.ndim for a in axes]):
                        grad = np.expand_dims(grad, axis=ax)

            grad = np.broadcast_to(grad, self.data.shape)

            self.grad += grad / count

        out._backward = _backward

        return out

    # def sum(self):
    #     out = Tensor(
    #         self.data.sum(),
    #         requires_grad=self.requires_grad,
    #         parents=(self,),
    #         op="Sum",
    #     )
    #
    #     def _backward():
    #         if self.requires_grad:
    #             self.grad += out.grad * np.ones_like(self.data)
    #
    #     out._backward = _backward
    #     return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Sum",
        )

        def _backward():
            if not self.requires_grad:
                return

            grad = out.grad

            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted([a if a >= 0 else a + self.data.ndim for a in axes]):
                    grad = np.expand_dims(grad, axis=ax)

            grad = np.broadcast_to(grad, self.data.shape)

            self.grad += grad

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

    def relu(self):
        from src.activations.ReLU import ReLU
        return ReLU.forward(self)
        # out = Tensor(
        #     np.maximum(0, self.data),
        #     requires_grad=self.requires_grad,
        #     parents=(self,),
        #     op="ReLU",
        # )
        #
        # def _backward():
        #     if self.requires_grad:
        #         self.grad += out.grad * (self.data > 0)
        #
        # out._backward = _backward
        #
        # return out

    def sigmoid(self):
        from src.activations import Sigmoid

        return Sigmoid.forward(self)

    def tanh(self):
        from src.activations import Tanh
        return Tanh.forward(self)

    def log(self):
        from src.math.log import Log
        return Log.forward(self)

    def exp(self):
        from src.math.exp import Exp
        return Exp.forward(self)

    # def softmax(self):
    #     from src.activations import Softmax
    #     return Softmax.forward(self)

    def softmax(self, axis=-1):
        from src.activations.Softmax import Softmax
        return Softmax.forward(self, axis=axis)

    def clip(self, min_value, max_value):
        from src.math.clip import Clip
        return Clip.forward(self, min_value, max_value)

    def argmax(self, axis=None):
        return Tensor(
            np.argmax(self.data, axis=axis),
            requires_grad=False,
        )

    def item(self):
        return self.data.item()

    def transpose(self, *axes):

        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Transpose",
        )

        def _backward():
            if not self.requires_grad:
                return

            inverse = np.argsort(axes)

            self.grad += np.transpose(
                out.grad,
                inverse,
            )

        out._backward = _backward

        return out

    def permute(self, *dims):
        return self.transpose(*dims)

    def reshape(self, *shape):

        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Reshape",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(
                    self.shape
                )

        out._backward = _backward

        return out

    def squeeze(self, axis=None):

        out = Tensor(
            np.squeeze(self.data, axis),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Squeeze",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(
                    self.shape
                )

        out._backward = _backward

        return out

    def unsqueeze(self, axis):

        out = Tensor(
            np.expand_dims(self.data, axis),
            requires_grad=self.requires_grad,
            parents=(self,),
            op="Unsqueeze",
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.squeeze(
                    out.grad,
                    axis=axis,
                )

        out._backward = _backward

        return out
    
    def backward(self, grad=None, release=False):
        topo = _topological_order(self)

        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float32).copy()

        for t in reversed(topo):
            t._bw()

            if release and t.parents:
                t._bw = _noop
                t.parents = ()
                t._grad = None

    @classmethod
    def arange(
            cls,
            start,
            stop=None,
            step=1,
            requires_grad=False,
    ):
        if stop is None:
            start, stop = 0, start

        return cls(
            np.arange(start, stop, step, dtype=np.float32),
            requires_grad=requires_grad,
        )

    @classmethod
    def random(cls, shape, requires_grad=False):
        return cls(
            np.random.random(shape).astype(np.float32),
            requires_grad=requires_grad,
        )