import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

plays = pd.read_csv("Shakespeare_data.csv")
# print("Shape of dataset: ",plays.shape)
# print("Columns in the dataset: ",plays.columns)

samples_lines = plays['PlayerLine'].sample(n=100, random_state=42)
data = " ".join(samples_lines)
# print(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
word_index = tokenizer.word_index
total_words = len(word_index) + 1

input_sequences = []
token_list = tokenizer.texts_to_sequences([data])[0]
for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 1000, input_length=max_sequence_len-1))
model.add(SimpleRNN(200))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=5, batch_size=64, verbose=1)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

seed_text = "That hath deprived me of your"
next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print(f"Next word prediction: {next_word}")