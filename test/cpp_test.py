import numpy as np
from cpp.build import _tensorforge_cpp

A = np.random.randn(4, 5).astype(np.float32)
B = np.random.randn(5, 3).astype(np.float32)

C = _tensorforge_cpp.matmul(A, B)

print(C)