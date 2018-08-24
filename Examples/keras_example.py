from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# https://keras.io/getting-started/sequential-model-guide
def create_sample_model():
    # Generate dummy data
    data = np.random.random((1000, 10))
    labels = np.random.randint(2, size=(1000, 1))

    # For a single-input model with 2 classes (binary classification):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=10))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
    return (data, labels, model)