import numpy as np

# class Tensor:
#     def __init__(self, shape):
#         self.shape = shape
#         self.data = None
#         self.init()
#
#     def init(self):
#         self.data = np.zeros(self.shape)
#
#     def zeros(self):
#         self.data = np.zeros(self.shape)
#
#     def random(self):
#         self.data = np.random.randn(*self.shape) * 0.01

# x = Tensor([[1, 2],
#             [3, 4]], requires_grad=True)
#
# print(x.shape)

class Tensor:

    def __init__(
            self,
            data,
            requires_grad=False,
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

    def backward(self):
        pass