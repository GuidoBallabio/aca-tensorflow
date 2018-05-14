"""This provides utilities to download, init and load cifar10 dataset."""

from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras


MODELS_DIR = Path(__file__).parent.parent / 'models'


def model_to_estimator(model):
    """
    Create an Estimator from the compiled Keras model.

    Note: 
        The initial model state of the keras model is preserved in the created 
        Estimator.
    """

    return keras.estimator.model_to_estimator(
            keras_model=model,
            model_dir=(MODELS_DIR / model.name).absolute()
            )


def convert_input_to_est_format(x, t, names, epochs):
    """
    Convert input to estimator format given model.input_names and epochs.

    To train, we call Estimator's train function:
    est_inception_v3.train(input_fn=train_input_fn, steps=2000)
    """

    return tf.estimator.inputs.numpy_input_fn(
        x={names[0]: x},
        y=t,
        num_epochs=epochs,
        shuffle=False)


def save_model(model):
    """Save keras model in default dir."""

    return model.save(MODELS_DIR / (model.name + '.h5'))


def load_model(name):
    """Save keras model in default dir."""

    filepath = MODELS_DIR / (name + '.h5')

    if filepath.exists():
        return keras.models.load_model(filepath)

    return None
