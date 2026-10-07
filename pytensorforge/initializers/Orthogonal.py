import numpy as np


class Orthogonal:

    def __call__(self, shape):

        rows, cols = shape

        a = np.random.randn(rows, cols)

        q, r = np.linalg.qr(a)

        return q.astype(np.float32)