import numpy as np
import tensorflow as tf
from tensorflow import keras

from cnn.utils import load_dataset_as_tensors, preprocess_dataset

BATCH_SIZE = 64

model = keras.Sequential(
    name='dense_cnn',
    layers=[
        keras.layers.Conv2D(
            32,
            5,
            input_shape=(32, 32, 3),
            padding='same',
            activation=tf.nn.relu),
        keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        keras.layers.Conv2D(64, 5, padding='same', activation=tf.nn.relu),
        keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy', keras.metrics.mean_squared_error])

if __name == '__main__':
    x_train, t_train, x_test, t_test = preprocess_dataset(load_dataset_as_tensors())

    hist = model.fit(
        x=x_train,
        y=t_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_split=0.2,
        shuffle=False,
        initial_epoch=0)

    print(hist.history)
