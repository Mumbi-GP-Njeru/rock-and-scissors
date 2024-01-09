!pip install tensorflow

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical

# Load the data
with open('smsspamcollection/smsspamcollection.tsv', 'r') as f:
    data = f.readlines()

# Split the data into messages and labels
messages = [line.split('\t')[0] for line in data]
labels = [line.split('\t')[1].strip() for line in data]

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(messages)

sequences = tokenizer.texts_to_sequences(messages)
data = pad_sequences(sequences)

# Split the data into training and validation sets
train_data = data[:1800]
train_labels = labels[:1800]

val_data = data[1800:]
val_labels = labels[1800:]

# Build the neural network model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=data.shape[1]))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert the labels to categorical format
train_labels = to_categorical(train_labels, num_classes=2)
val_labels = to_categorical(val_labels, num_classes=2)

# Train the model
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10)

# Create the predict_message function
def predict_message(message):
    sequence = tokenizer.texts_to_sequences([message])
    data = pad_sequences(sequence, maxlen=data.shape[1])
    prediction = model.predict(data)
    probability = prediction[0][1]
    label = 'spam' if probability > 0.5 else 'ham'
    return [probability, label]
