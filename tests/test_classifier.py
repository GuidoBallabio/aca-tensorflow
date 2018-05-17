"""Test TfClassifier"""

import numpy as np
import tensorflow as tf

from cnn.model_class import TfClassifier, HALF_MAX_BATCH_SIZE


def fake_tfclassifier():
    def fp_fn(train_mode, keep_prob_placeholder):
        inputs = tf.placeholder(
            tf.float32, shape=[None, 2, 2, 2], name="features")
        flat = tf.layers.flatten(inputs)
        logits = tf.layers.dense(inputs=flat, units=10, activation=tf.nn.relu)
        return logits

    def loss_fn(logits):
        labels = tf.placeholder(tf.int32, name="labels")
        return tf.losses.sparse_softmax_cross_entropy(labels, logits)

    def eval_fn(predictions):
        labels = tf.get_default_graph().get_tensor_by_name("labels:0")
        eval_metrics = {
            "mse":
            tf.metrics.mean_squared_error(
                labels=tf.cast(labels, tf.float32),
                predictions=predictions["logits"])
        }

        return eval_metrics

    return TfClassifier('name', fp_fn, loss_fn, eval_fn,
                        tf.train.AdamOptimizer())


def test_split_and_batch():
    x_train = np.arange(8 * 4200).reshape(-1, 2, 2, 2)
    n_samples = x_train.shape[0]
    t_train = np.random.choice(np.arange(10), size=(n_samples, 1))

    inputs = [x_train, t_train]
    input_names = ["features", "labels"]
    batch_size = 2
    validation_split = 0.5

    model = fake_tfclassifier()

    with tf.Session(graph=model.train_ops_graph[1]) as s:
        split = model._split_and_batch(inputs, input_names, batch_size,
                                       validation_split, 0.5)
        input_tensors = [
            tf.get_default_graph().get_tensor_by_name(n + ':0')
            for n in input_names
        ]

    assert len(split) == 2

    train_LD = split[0]
    val_LD = split[1]

    train_samples = 0
    for d in train_LD:
        train_samples += d[input_tensors[0]].shape[0]
    train_samples

    val_samples = 0
    for d in val_LD:
        val_samples += d[input_tensors[0]].shape[0]
    val_samples

    assert n_samples == train_samples + val_samples

    assert len(train_LD) * batch_size == train_samples

    assert train_samples == np.int(n_samples * (1 - validation_split))

    assert val_samples == np.int(n_samples * validation_split)

    for d in train_LD:
        assert batch_size == d[input_tensors[0]].shape[0]

    for d in val_LD:
        assert HALF_MAX_BATCH_SIZE * 2 >= d[input_tensors[0]].shape[0]


def test_init_dict_split_max():
    x_train = np.arange(8 * 4200).reshape(-1, 2, 2, 2)
    n_samples = x_train.shape[0]
    t_train = np.random.choice(np.arange(10), size=(n_samples, 1))

    inputs = [x_train, t_train]
    input_names = ["features", "labels"]

    model = fake_tfclassifier()

    with tf.Session(graph=model.train_ops_graph[1]) as s:
        input_LD = model._init_dict_split_max(inputs, input_names)
        input_tensors = [
            tf.get_default_graph().get_tensor_by_name(n + ':0')
            for n in input_names
        ]

    out_samples = 0
    for d in input_LD:
        out_samples += d[input_tensors[0]].shape[0]
    out_samples

    assert n_samples == out_samples

    for d in input_LD:
        assert HALF_MAX_BATCH_SIZE * 2 >= d[input_tensors[0]].shape[0]
