# import numpy as np
#
# from src.neural.Dense import Dense
#
# dense = Dense(9, "relu")
#
# dense.build(input_shape=(None, 9))
# y = dense.call(np.array([[1,2],[3,4],[5,6]]))
#
# print(y)
from src.core.Tensor import Tensor
from src.models.seq.Sequential import Sequential
from src.neural.Dense import Dense

# X_train = Tensor([
#     [1, 2],
#     [1, 2],
# ])
#
# y_train = Tensor([
#     [2],
#     [2],
# ])
#
# model = Sequential()
#
# model.add(Dense(128, activation="relu"))
# model.add(Dense(64, activation="relu"))
# # model.add(Dense(10, activation="softmax"))
#
# model.compile(
#     optimizer="sgd",
#     loss="mse",
# )
#
# model.fit(X_train, y_train, epochs=20)


# y = 2x
# X_train = Tensor([
#     [1],
#     [2],
#     [3],
#     [4],
# ], requires_grad=False)
#
# y_train = Tensor([
#     [2],
#     [4],
#     [6],
#     [8],
# ], requires_grad=False)

# X_train = Tensor([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1],
# ], requires_grad=False)
#
# y_train = Tensor([
#     [0],
#     [1],
#     [1],
#     [0],
# ], requires_grad=False)
#
# model = Sequential()
#
# model.add(Dense(4, activation='relu'))
# model.add(Dense(1, activation="sigmoid"))
#
# model.compile(
#     optimizer="sgd",
#     loss="mse",
# )
#
# model.fit(X_train, y_train, epochs=1000, batch_size=4)
# model.summary()
#
# predictions = model.predict(X_train)
# print(predictions.data)

# x = Tensor([[1.0], [2.0]], requires_grad=True)
# y = Tensor([[2.0], [4.0]])
#
# loss = ((x - y) * (x - y)).mean()
#
# print("loss:", loss.data)
#
# loss.backward()
#
# print("x.grad:")
# print(x.grad)

# X_train = Tensor([[1.0], [2.0], [3.0], [4.0]])
# y_train = Tensor([[2.0], [4.0], [6.0], [8.0]])
#
# model = Sequential()
# model.add(Dense(1))
#
# model.compile(optimizer="sgd", loss="mse")
#
# model.fit(X_train, y_train, epochs=1000, batch_size=4)

from src.core.Tensor import Tensor
from src.models.seq.Sequential import Sequential
from src.neural.Dense import Dense

# Train
# X_train = Tensor([
#     [1],
#     [2],
#     [3],
# ], requires_grad=False)
#
# y_train = Tensor([
#     [2],
#     [4],
#     [6],
# ], requires_grad=False)
#
# # Test
# X_test = Tensor([
#     [4],
# ], requires_grad=False)
#
# y_test = Tensor([
#     [8],
# ], requires_grad=False)

# y = 2x
X_train = Tensor([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
], requires_grad=False)

y_train = Tensor([
    [2],
    [4],
    [6],
    [8],
    [10],
    [12],
    [14],
    [16],
], requires_grad=False)

X_test = Tensor([
    [9],
    [10],
], requires_grad=False)

y_test = Tensor([
    [18],
    [20],
], requires_grad=False)

model = Sequential()

model.add(Dense(1))

model.compile(
    optimizer="sgd",
    loss="mse",
)

model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=3,
)

prediction = model.predict(X_test)

print("Prediction:")
print(prediction.data)

print("Expected:")
print(y_test.data)