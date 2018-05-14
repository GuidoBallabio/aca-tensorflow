import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import load_dataset_as_tensors, preprocess_dataset, save_keras_model, load_keras_model

BATCH_SIZE = 64
NET_NAME = 'dense_cnn'

def model_fn():
    """Define and return dense cnn model, by keras."""

    model = keras.Sequential(
        name=NET_NAME,
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

    return model


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = preprocess_dataset(
        *load_dataset_as_tensors())

    '''Normalization of data'''
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    '''Model load, train and save'''
    model = load_keras_model(NET_NAME)
    if model is None:
        model = model_fn()

    hist = model.fit(
        x=x_train,
        y=t_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        validation_split=0.2,
        initial_epoch=0)

    save_keras_model(model)

    print(hist.history)
