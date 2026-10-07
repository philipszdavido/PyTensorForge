from pytensorforge.core.Tensor import Tensor
from pytensorforge.models.seq.Sequential import Sequential
from pytensorforge.neural.Dense import Dense

X_train = Tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], requires_grad=False)

y_train = Tensor([
    [0],
    [1],
    [1],
    [0],
], requires_grad=False)

model = Sequential()

model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

model.fit(X_train, y_train, epochs=5000)
model.summary()

prediction = model.predict(X_train)

print("Prediction:")
print(prediction.data)
