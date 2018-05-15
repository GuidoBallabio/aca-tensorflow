"""This module defines a Simple CNN"""

import numpy as np
import tensorflow as tf

from cnn.utils.dataset import load_cifar10, dataset_preprocessing_by_keras
from cnn.model_class import TfClassifier

BATCH_SIZE = 64
NET_NAME = 'dense_cnn'
EPOCHS = 50


def forward_pass(train_mode_placeholder=None):
    features = tf.placeholder(
        tf.float32, shape=(None, 32, 32, 3), name="features")

    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)

    pool2_flat = tf.layers.flatten(pool2)

    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=train_mode_placeholder)

    return tf.layers.dense(inputs=dropout, units=10, name="logits")


def loss_fn(logits):
    labels = tf.placeholder(tf.float32, [None, 10], name="labels")
    return tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


def eval_fn(predictions):
    labels = tf.get_default_graph().get_tensor_by_name("labels:0")
    eval_metrics = {
        "accuracy":
        tf.metrics.accuracy(
            labels=tf.argmax(labels, axis=1),
            predictions=predictions["classes"]),
        "mse":
        tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions["logits"])
    }
    tf.summary.scalar("accuracy", eval_metrics["accuracy"][0])
    tf.summary.scalar("mse", eval_metrics["mse"][0])
    return eval_metrics


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = load_cifar10()

    x_train = dataset_preprocessing_by_keras(x_train)

    model = TfClassifier(NET_NAME, forward_pass, loss_fn, eval_fn,
                         tf.train.AdamOptimizer())
    history = model.fit(
        [x_train, t_train],
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        epochs=EPOCHS,
        verbosity=1)

    print(history)

    evals = model.evaluate([x_test, t_test])

    print(evals)
