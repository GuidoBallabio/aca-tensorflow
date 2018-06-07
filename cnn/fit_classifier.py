"""This module defines fitter class that implements a command line"""

import argparse

import tensorflow as tf

from cnn.model_class import TfClassifier
from cnn.utils.dataset import dataset_preprocessing_by_keras, load_cifar10
from cnn.utils.graph_manipulation import write_graph

BATCH_SIZE = 32
EPOCHS = 50
DROP_PROB = 0.5
VALIDATION_SPLIT = 0.2


class FitOnCIFAR10:
    def __init__(self, forward_pass, loss_fn, eval_fn):
        self.forward_pass = forward_pass
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

    def setup_data(self):
        x_train, t_train, x_test, t_test = load_cifar10()
        x_train = dataset_preprocessing_by_keras(x_train)
        return x_train, t_train, x_test, t_test

    def train_and_save(self, name, batch_size, epochs, drop_prob,
                       validation_split, quantization, verbosity):

        self.model = TfClassifier(
            name,
            self.forward_pass,
            self.loss_fn,
            self.eval_fn,
            tf.train.AdamOptimizer(),
            quantization=quantization)

        x_train, t_train, x_test, t_test = self.setup_data()

        self.model.fit(
            [x_train, t_train],
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs,
            verbosity=verbosity,
            drop_prob=drop_prob)

        evals = self.model.evaluate([x_test, t_test])

        self.model.save_optimazed_graph()

    def main(self):
        parser = argparse.ArgumentParser(description="Fit a net")
        parser.add_argument("name", type=str, help="The name of th net")
        parser.add_argument(
            "--batch_size",
            action="store",
            type=int,
            default=BATCH_SIZE,
            help="Batch size")
        parser.add_argument(
            "--epochs",
            action="store",
            type=int,
            default=EPOCHS,
            help="Number of epochs")
        parser.add_argument(
            "--drop_prob",
            action="store",
            type=float,
            default=DROP_PROB,
            help="Dropout probability")
        parser.add_argument(
            "--validation_split",
            action="store",
            type=float,
            default=VALIDATION_SPLIT,
            help="Percentage for validation")
        parser.add_argument(
            "--quantization", action="store_true", help="Enable quantization")
        parser.add_argument(
            "--verbosity", "-v", action="count", help="Set verbosity level")

        args = parser.parse_args()

        self.train_and_save(**vars(args))
