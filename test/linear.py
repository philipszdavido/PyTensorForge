from src.models.regression.Linear import Linear
import numpy as np

# 5 samples, 3 features
X = np.array([
    [1000, 2, 10],
    [1500, 3, 5],
    [1800, 3, 15],
    [2400, 4, 2],
    [3000, 4, 20]
], dtype=np.int32)

y = np.array([200, 290, 310, 400, 510], dtype=np.int32)

linear = Linear((3,), (1,))

# print(linear.W.data)
# print(linear.b.data)

for _ in range(30):
    for x, target in zip(X, y):
        predicted = linear.forward(x)

        # backward
        linear.backward(target, x, predicted)

        linear.update(1e-7)

for x, target in zip(X, y):
    print(linear.forward(x), target)