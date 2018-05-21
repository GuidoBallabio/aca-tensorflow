"""Test TfClassifier"""

import numpy as np
import tensorflow as tf

from cnn.model_class import TfClassifier, HALF_MAX_BATCH_SIZE


def fake_tfclassifier():
    def fp_fn(train_mode, drop_prob_placeholder):
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
    validation_split = 0.2

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

    for d in train_LD:
        assert d["drop_prob:0"] == 0.5

    for d in val_LD:
        assert d["drop_prob:0"] == 0.0


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


def test_split_data_dict_in_perc():
    model = fake_tfclassifier()
    a = np.arange(1, 11)
    n_samples = len(a)
    percs = np.array([0.72, 0.84])

    dic = {'a': a, 'b': a}
    output_LD = model._split_data_dict_in_perc(dic, n_samples, percs)

    lens = np.array([x['a'].shape[0] for x in output_LD])
    splits_index = (n_samples * np.append(percs, 1.0)).astype(np.int)

    assert n_samples == lens.sum()

    assert np.all(lens.cumsum() == splits_index)

    assert a[0] == output_LD[0]['a'][0]


def test_batch_data_dict():
    model = fake_tfclassifier()
    a = np.arange(10)
    n_samples = len(a)
    batch_size = 2

    dic = {'a': a, 'b': a}
    output_LD = model._batch_data_dict(dic, n_samples, batch_size)

    n_of_dicts = np.int(len(a) / batch_size)
    n_elem_in_first_dict = len(output_LD[0]['a'])

    assert len(output_LD) == n_of_dicts

    assert np.isclose(n_elem_in_first_dict, batch_size, atol=1)


def test_set_drop_prob_to_LD():
    model = fake_tfclassifier()
    a = np.arange(2)
    drop_prob = 0.4

    list_dicts = [{'a': a, 'b': a}, {'c': a, 'drop_prob:0': 0.3}]
    output_LD = model._set_drop_prob_to_LD(list_dicts, drop_prob)

    assert len(output_LD) == len(list_dicts)

    assert len(output_LD[0].keys()) == len(list_dicts[0].keys())

    assert output_LD[0]['drop_prob:0'] == drop_prob

    assert output_LD[1]['drop_prob:0'] == drop_prob


def test_init_dict():
    model = fake_tfclassifier()
    inputs = [1, 1]
    input_names = ['labels', 'drop_prob']
    with tf.Session(graph=model.train_ops_graph[1]) as s:
        tensors, output_LD = model._init_dict(inputs, input_names)

    for t in tensors:
        assert output_LD[t] == 1
