"""General function for formatting inputs"""

import tensorflow as tf
import numpy as np

HALF_MAX_BATCH_SIZE = 2000


def init_dict(inputs, input_names):
    """
    Example:
        inputs = [1,1]
        input_names = ['labels','drop_prob']
    Outputs: 
        input_tensors = [<labels' tensor>, <drop_prob's tensor>]
        input_DL = {<labels' tensor> : 1, <drop_prob's tensor> : 1}
    """
    input_names = [n + ':0' for n in input_names]

    input_DL = dict(zip(input_names, inputs))  # Dict from pairs of list

    return input_names, input_DL


def split_data_dict_in_perc(input_dict, n_samples, percs):
    """
    Example:
        input_dict = {'a':[1,2,3,4], 'b':[5,6,7,8]}
        n_samples = 4
        percs = np.array([0.75])
    Output: [{'a':[1,2,3], 'b':[5,6,7]},
             {'a':[4], 'b':[8]}]
    """
    for k, v in input_dict.items():
        # Here n_sample == len(v)
        input_dict[k] = np.split(v, (n_samples * percs).astype(np.int))

    input_LD = [dict(zip(input_dict, t))
                for t in zip(*input_dict.values())]  # List of Dicts

    return input_LD


def batch_data_dict(input_dict, n_samples, batch_size):
    """
    Example:
        input_dict = {'a':[1,2,3,4,5,6], 'b':[5,6,7,8,9,0]}
        n_samples = 4
        batch_size = 2
    Output: [{'a':[1,2], 'b':[5,6]},
             {'a':[3,4], 'b':[7,8]},
             {'a':[5,6], 'b':[9,0]}]
    """
    n_batches, drop = np.divmod(n_samples, batch_size)

    if n_batches == 0:
        n_batches = 1

    for k, v in input_dict.items():
        input_dict[k] = np.array_split(v, n_batches)

    out_LD = [dict(zip(input_dict, t)) for t in zip(*input_dict.values())]

    return out_LD


def init_dict_split_max(inputs, input_names):
    """
    Example:
        inputs = [4200 features 2x2x2, 4200 labels]
        input_names = ['features','labels']
    Output: [{<features' tensor> : 2200 features 2x2x2, <labels' tensor> :
        2200 labels 2x2x2},
        {<features' tensor> : 2000 features 2x2x2, <labels' tensor> : 
        2000 labels 2x2x2}]
    """
    input_dict = init_dict(inputs, input_names)[1]
    n_samples = inputs[0].shape[0]

    out_LD = batch_data_dict(input_dict, n_samples, HALF_MAX_BATCH_SIZE)

    return out_LD


def split_and_batch(inputs, input_names, batch_size, validation_split):
    """
    Example:
        inputs = [4200 features 2x2x2, 4200 labels]
        input_names = ['features','labels']
        batch_size = 2
        validation_split = 0.2
        drop_prob = 0.5
    Output: 
        train_LD = [{<features' tensor> : 2 features 2x2x2, <labels' tensor>
                     : 2 labels, <drop_prob's tensor> : 0.5},
                    {<features' tensor> : 2 features 2x2x2, <labels' tensor>
                     : 2 labels, <drop_prob's tensor> : 0.5},
                    ... until features stored are 0.8 * 4200 times]
                    val_LD = [{<features' tensor> : 2000 features 2x2x2, 
                    <labels' tensor> : 2000 labels, <drop_prob's tensor> : 1},
                    ... until features stored are 0.2 * 4200 times]
        in this case val_LD = [{<features' tensor> : 840 features 2x2x2, 
        <labels' tensor> : 840 labels, <drop_prob's tensor> : 1}]
    """
    n_samples = inputs[0].shape[0]

    input_names, input_DL = init_dict(inputs, input_names)

    input_LD = split_data_dict_in_perc(input_DL, n_samples,
                                       np.array([1 - validation_split]))

    train_dict = input_LD[0]
    val_dict = input_LD[1]

    n_train_samples = train_dict[input_names[0]].shape[0]

    train_LD = batch_data_dict(train_dict, n_train_samples, batch_size)

    val_LD = batch_data_dict(val_dict, n_samples - n_train_samples,
                             HALF_MAX_BATCH_SIZE)

    return train_LD, val_LD
