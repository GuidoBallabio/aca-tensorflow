"""This module defines a Dense CNN"""

import numpy as np
import tensorflow as tf

from cnn.fit_classifier import FitOnCIFAR10

NET_NAME = 'dense'


def forward_pass(train_mode, drop_prob_placeholder):
    features = tf.placeholder(
        tf.float32, shape=(None, 32, 32, 3), name="features")

    conv1 = tf.layers.conv2d(
        inputs=features,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)

    pool1_flat = tf.layers.flatten(pool1)

    dense1 = tf.layers.dense(
        inputs=pool1_flat, units=512, activation=tf.nn.relu)

    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense2, rate=drop_prob_placeholder, training=train_mode)

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
    fitter = FitOnCIFAR10(forward_pass, loss_fn, eval_fn)
    fitter.main()
