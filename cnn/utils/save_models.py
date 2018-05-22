"""This provides utilities to download, init and load cifar10 dataset."""

from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

MODELS_DIR = Path(__file__).parent.parent / 'models'


def transform_graph(graph_def, input_names, output_names, transforms):
    out_graph_def = TransformGraph(graph_def, input_names, output_names,
                                   transforms)

    out_graph = tf.Graph()
    with out_graph.as_default():
        tf.import_graph_def(out_graph_def, name='')

    return out_graph


def write_graph(graph, name, dir_path=MODELS_DIR.as_posix()):
    tf.train.write_graph(graph.as_graph_def(), dir_path, name, as_text=False)


def load_frozen_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    # Then, we import the graph_def into a new Graph and returns it
    with graph.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')

    return graph


def predict_from_frozen(graph, inputs, input_names, output_names):

    input_LD = init_dict_split_max(inputs, input_names)

    with tf.Session(graph=graph) as sess:
        out = []
        for input_dict in input_LD:
            out.append(sess.run(ops, feed_dict=input_dict))

    return out


def model_to_estimator(model):
    """
    Create an Estimator from the compiled Keras model.

    Note: 
        The initial model state of the keras model is preserved in the created 
        Estimator.
    """

    return keras.estimator.model_to_estimator(
        keras_model=model, model_dir=(MODELS_DIR / model.name).absolute())


def convert_input_to_est_format(x, t, names, epochs):
    """
    Convert input to estimator format given model.input_names and epochs.

    To train, we call Estimator's train function:
    est_inception_v3.train(input_fn=train_input_fn, steps=2000)
    """

    return tf.estimator.inputs.numpy_input_fn(
        x={names[0]: x}, y=t, num_epochs=epochs, shuffle=False)


def save_model(model):
    """Save keras model in default dir."""

    return model.save(MODELS_DIR / (model.name + '.h5'))


def load_model(name):
    """Save keras model in default dir."""

    filepath = MODELS_DIR / (name + '.h5')

    if filepath.exists():
        return keras.models.load_model(filepath)

    return None
