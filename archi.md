```
Tensor
    ├── data
    ├── grad
    ├── requires_grad
    ├── backward()

Parameter (inherits Tensor)

Layer
    ├── parameters()
    ├── add_weight()
    ├── build()
    ├── call()
    ├── __call__()

Dense
Conv2D
BatchNorm
Dropout
Embedding
Flatten
LSTM
GRU

Sequential
Model

Losses
    ├── MSE
    ├── CrossEntropy
    ├── BCE

Optimizers
    ├── SGD
    ├── Momentum
    ├── RMSProp
    ├── Adam
    ├── AdamW

Initializers
Activations
Datasets
DataLoader
```