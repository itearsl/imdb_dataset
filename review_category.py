import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

def vectorise_data(data, vector_len = 10000):
    matrix = np.zeros((len(data), vector_len))
    for i, secuence in enumerate(data):
        matrix[i, secuence] = 1
    return matrix

train_data = vectorise_data(train_data)
test_data = vectorise_data(test_data)

train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer='rmsprop')

v_data = test_data[:10000]
v_labels = test_labels[:10000]


history = model.fit(
    train_data,
    train_labels,
    epochs= 3,
    batch_size=128,
    validation_data=(v_data, v_labels)
)

history = history.history
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.show()