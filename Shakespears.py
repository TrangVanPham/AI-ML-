# Import necessary libraries
import numpy as np
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.utils import to_categorical

# Step 1: Read the text
with open('data.txt', 'r') as file:
    text = file.read()
    lines = text.lower().split('\n')

# Step 2: Tokenization and sequence preparation
words = text_to_word_sequence(text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
vocabulary_size = len(tokenizer.word_index) + 1

# Convert lines to sequences
sequences = tokenizer.texts_to_sequences(lines)

# Step 3: Create subsequences
subsequences = []
for sequence in sequences:
    for i in range(1, len(sequence)):
        subsequence = sequence[:i + 1]
        subsequences.append(subsequence)

# Step 4: Padding sequences
sequence_length = max([len(sequence) for sequence in subsequences])
sequences = pad_sequences(subsequences, maxlen=sequence_length, padding='pre')

# Step 5: Encode target labels
x, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocabulary_size)

# Step 6: Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=vocabulary_size, output_dim=100, input_length=sequence_length - 1))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(vocabulary_size, activation='softmax'))

# Step 7: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the model
model.fit(x, y, epochs=500, verbose=1)

# Step 9: Return the model
print("Model Training Complete!")
