import pickle
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_DIR = Path(__file__).parent.parent / 'data'
DATASET_FILE = 'cifar10.h5'


def download_and_extract():
    """Download and extract cifar10 dataset."""

    dest_directory = DATA_DIR
    extracted_dir = dest_directory / 'cifar-10-batches-py'

    if not dest_directory.exists():
        dest_directory.mkdir()

    filepath = (dest_directory / DATA_URL.split('/')[-1])

    if (DATA_DIR / DATASET_FILE).exists():
        return

    if not filepath.exists() and not extracted_dir.exists():
        filepath, _ = urlretrieve(DATA_URL, filepath.absolute())
        print('Successfully downloaded')

    if not extracted_dir.exists():
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        filepath.unlink()
        print('Extracted dataset')


def _load_arrays_from_pickles(files):
    """Load shuffled data from pickle files as numpy tensors concateneted."""

    data = []
    labels = []

    for batch in files:
        with open(DATA_DIR / 'cifar-10-batches-py' / batch, 'rb') as f_b:
            d = pickle.load(f_b, encoding='latin')
        data.append(d['data'])
        labels.append(np.array(d['labels']))

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    length = len(labels)

    data, labels = shuffle(data, labels, random_state=0)

    return data.reshape(length, 3, 32, 32), labels


def convert_dataset_to_h5():
    """Convet cifar10 dataset to h5 format."""

    h5_filename = DATA_DIR / DATASET_FILE

    if not h5_filename.exists():
        x, t = load_arrays_from_pickles(
            ("data_batch_{}".format(i) for i in range(1, 6)))
        x_test, t_test = _load_arrays_from_pickles(["test_batch"])

        comp_kwargs = {'compression': 'gzip'}

        with h5py.File(h5_filename.absolute(), 'w') as f:
            f.create_dataset('data/train', data=x, **comp_kwargs)
            f.create_dataset(
                'label/train', data=t.astype(np.int_), **comp_kwargs)
            f.create_dataset('data/test', data=x_test, **comp_kwargs)
            f.create_dataset(
                'label/test', data=t_test.astype(np.int_), **comp_kwargs)

        print('Conversion to HDF5 file successful')


def load_dataset_as_tensors():
    """Load dataset from h5 (not lazily)."""

    h5_filename = DATA_DIR / DATASET_FILE
    with h5py.File(h5_filename.absolute(), 'r') as ds:
        return (ds['data/train'][()], ds['label/train'][()],
                ds['data/test'][()], ds['label/test'][()])


def preprocess_dataset(x_train, t_train, x_test, t_test, NHWC):
    """Process data: label -> one hot encoding, transpose if in NHWC order."""

    x_train = x_train.astype(np.float)
    x_test = x_test.astype(np.float)
    t_train = keras.utils.to_categorical(t_train)
    t_test = keras.utils.to_categorical(t_test)

    if NHWC:
        return (x_train.transpose(0, 2, 3, 1), t_train,
                x_test.transpose(0, 2, 3, 1), t_test)
    else:
        return (x_train, t_train, x_test, t_test)


def load_cifar10(NHWC=True):
    """Load cifar10 dataset as np.ndarray ready to go."""

    return preprocess_dataset(*load_dataset_as_tensors(), NHWC)


def dataset_preprocessing_by_keras(x_train):
    """Dataset with preprocessing by Keras.
    """

    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(x_train)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)

    return datagen.standardize(x_train)


if __name__ == '__main__':
    download_and_extract()
    convert_dataset_to_h5()
