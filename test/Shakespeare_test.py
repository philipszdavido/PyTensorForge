import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.core.Tensor import Tensor
from src.models.seq.Sequential import Sequential
from src.models.embedding.Embedding import Embedding
from src.neural.RNN import RNN
from src.neural.Dense import Dense
from src.activations.Softmax import Softmax


##################################################
# Load Shakespeare
##################################################

plays = pd.read_csv("Shakespeare_data.csv")

samples = plays["PlayerLine"].sample(
    n=100,
    random_state=42,
)

text = " ".join(samples.astype(str))


##################################################
# Tokenize
##################################################

tokenizer = Tokenizer()

tokenizer.fit_on_texts([text])

vocab = tokenizer.word_index

vocab_size = len(vocab) + 1


##################################################
# Build training sequences
##################################################

token_list = tokenizer.texts_to_sequences([text])[0]

sequences = []

for i in range(1, len(token_list)):
    sequences.append(token_list[: i + 1])

max_len = max(len(x) for x in sequences)

sequences = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="pre",
)


##################################################
# Inputs
##################################################

X = sequences[:, :-1]

##################################################
# Labels
##################################################

y = sequences[:, -1]

##################################################
# One-hot encode
##################################################

Y = np.zeros((len(y), vocab_size), dtype=np.float32)

Y[np.arange(len(y)), y] = 1.0


##################################################
# Convert to Tensor
##################################################

X_train = Tensor(
    X,
    requires_grad=False,
)

Y_train = Tensor(
    Y,
    requires_grad=False,
)


##################################################
# Model
##################################################

model = Sequential()

model.add(
    Embedding(
        vocab_size=vocab_size,
        embedding_dim=128,
    )
)

model.add(
    RNN(
        hidden_size=256,
    )
)

model.add(
    Dense(vocab_size, activation="softmax")
)

# model.add(
#     Softmax()
# )

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
)

model.summary()

model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=64,
)

def predict_next_word(model, tokenizer, text, max_len):

    tokens = tokenizer.texts_to_sequences([text])[0]

    tokens = pad_sequences(
        [tokens],
        maxlen=max_len - 1,
        padding="pre",
    )

    x = Tensor(tokens, requires_grad=False)

    probs = model.predict(x)

    probs = probs.data

    idx = np.argmax(probs)

    for word, i in tokenizer.word_index.items():
        if i == idx:
            return word

    return "<UNK>"

seed = "That hath deprived me of your"

print(predict_next_word(
    model,
    tokenizer,
    seed,
    max_len,
))