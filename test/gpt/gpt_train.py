import numpy as np

from pytensorforge.core.Tensor import Tensor
from pytensorforge.models.embedding.Embedding import Embedding
from pytensorforge.models.seq.Sequential import Sequential
from pytensorforge.models.transformers.LastToken import LastToken
from pytensorforge.models.transformers.TransformerBlock import TransformerBlock
from pytensorforge.neural.Dense import Dense
from pytensorforge.optimizers import Adam

####################################################
# Dataset
####################################################

text = """

<BOS> the cat sat on the mat <EOS>
<BOS> the dog sat on the rug <EOS>
<BOS> the cat chased the mouse <EOS>
<BOS> the dog chased the cat <EOS>

<BOS> User: What is Python?
 Assistant: Python is a programming language <EOS>

<BOS> User: Who wrote Romeo and Juliet? 
Assistant: William Shakespeare wrote Romeo and Juliet <EOS>

<BOS> User: Explain recursion
 Assistant: Recursion is a technique where a function calls itself <EOS>
"""

words = text.lower().split()

print(words)

vocab = sorted(set(words))

word_to_index = {
    w: i
    for i, w in enumerate(vocab)
}

index_to_word = {
    i: w
    for w, i in word_to_index.items()
}

sequence_length = 3

X = []
y = []

for i in range(len(words) - sequence_length):

    X.append([
        word_to_index[w]
        for w in words[i:i + sequence_length]
    ])

    y.append(
        word_to_index[
            words[i + sequence_length]
        ]
    )

X_train = Tensor(
    np.array(X),
    requires_grad=False,
)

y_train = Tensor(
    np.array(y),
    requires_grad=False,
)

####################################################
# Model
####################################################

# model = Sequential()
#
# model.add(
#     Embedding(
#         vocab_size=len(vocab),
#         embedding_dim=256,
#     )
# )
#
# for _ in range(6):
#
#     model.add(
#         TransformerBlock(
#             d_model=256,
#             num_heads=8,
#             ff_dim=1024,
#         )
#     )
#
# model.add(LastToken())
#
# model.add(
#     Dense(
#         units=len(vocab),
#         activation="softmax",
#     )
# )

model = Sequential()

model.add(
    Embedding(
        vocab_size=len(vocab),
        embedding_dim=64,
    )
)

model.add(
    TransformerBlock(
        d_model=64,
        num_heads=4,
        ff_dim=128,
    )
)

model.add(LastToken())

model.add(
    Dense(
        len(vocab),
        #activation="softmax",
    )
)

model.compile(
    optimizer=Adam(lr=1e-4),
    loss="cross_entropy_with_logits",
)

model.summary()

model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=4,
)

####################################################
# Save
####################################################

metadata = {
    "word_to_index": word_to_index,
    "index_to_word": index_to_word,
    "sequence_length": sequence_length,
    "vocab_size": len(vocab),
}

model.save(
    "transformer.ptf",
    metadata,
)