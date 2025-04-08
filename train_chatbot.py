import tensorflow as tf
import numpy as np
import pickle
from preprocessing import preprocess_text, encode_labels
from sklearn.feature_extraction.text import CountVectorizer

# Load training data from file
def load_training_data(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    training_data = []
    for line in data:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            text, label = parts
            training_data.append({"text": text, "label": label})
    return training_data

# Load training data
training_data = load_training_data("training_data.txt")

# Preprocess the text
texts = [preprocess_text(data["text"]) for data in training_data]
labels = [data["label"] for data in training_data]

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Encode the labels
label_encoder, y = encode_labels(labels)

# LSTM neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=X.shape[1], output_dim=128, input_length=X.shape[1]),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with epochs
model.fit(X, y, epochs=500, batch_size=8)

# Save the model and vectorizer
model.save("chatbot_model.h5")
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open("label_encoder.pkl", "wb") as label_encoder_file:
    pickle.dump(label_encoder, label_encoder_file)