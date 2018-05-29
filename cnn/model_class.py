"""Module implementig TfClassifier class."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from cnn.utils.graph_manipulation import (MODELS_DIR, load_frozen_graph,
                                          transform_graph, write_graph,
                                          freeze_graph, optimize_for_inference)
from cnn.utils.prep_inputs import init_dict_split_max, split_and_batch


class TfClassifier:
    """Tensorflow-based classifier for NNs with simple API and quantization.

    This class wraps a forward pass part of a Neural Network graph as well as 
    the loss function and the optimizer, exposing a simple interface. 
    """

    def __init__(self,
                 name,
                 forward_pass_fn,
                 loss_fn,
                 eval_fn,
                 optimizer,
                 quantization=False):
        """Initializer of TfClassifier.

         Args:
            name (str): Name of the model and used for files written on disk.

            forward_pass_fn (function): Main model function: forward pass.
                This funcion must accept a single parameter of type
                tf.placeholder(tf.bool, shape=()) that enables training mode,

                if the parameter is set to True then the func is in training.
                The function must return logits as tensor.
                Placeholders must be used for inputs with names.
            loss_fn (function): Function of model that defines loss.
                This function must accept predictions as a tensor and return
                the loss as a tensor.
                Placeholders must be used for input labels with names.
            eval_fn (function): Description of `param3`.
            optimizer (function): Instance of a :class:`tf.Optimazer` subclass.
            quantization (bool, optional): If True enables fake quantization 
                and :method:`freeze` will enable to implement a real 
                quantization through the command line tool GraphTransform.
        """

        self.name = name
        self.forward_pass_fn = forward_pass_fn
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.optimizer = optimizer
        self.quantization = quantization
        self.train_ops_graph = self.train_graph()
        self.eval_ops_graph = self.evaluate_graph()
        self.predict_ops_graph = self.predict_graph()
        self.tb_path_train = (
            Path("/tmp/log-tb/") / self.name / "training").as_posix()
        self.tb_path_val = (
            Path("/tmp/log-tb/") / self.name / "validation").as_posix()
        self.save_path = MODELS_DIR / self.name / "model.ckpt"

    def _infer(self, train_mode=False):

        self.drop_prob_placeholder = tf.placeholder(
            tf.float32, (), name="drop_prob")

        logits = self.forward_pass_fn(train_mode, self.drop_prob_placeholder)

        predictions = {
            "logits": logits,
            "classes": tf.argmax(logits, axis=1, name="classes"),
            "probabilities": tf.nn.softmax(logits, name="softmax")
        }

        return predictions

    def _calculate_loss(self, logits):
        loss = self.loss_fn(logits=logits)
        tf.summary.scalar("loss", loss)

        return loss

    def _optimize(self, loss):
        return self.optimizer.minimize(loss)

    def _evaluate_op(self, predictions):
        return self.eval_fn(predictions)

    def train_graph(self):
        graph = tf.Graph()

        with graph.as_default() as g:
            predictions = self._infer(train_mode=True)
            loss = self._calculate_loss(predictions["logits"])
            if self.quantization:
                tf.contrib.quantize.create_training_graph(quant_delay=2000)
            train_op = self._optimize(loss)
            evals = self._evaluate_op(predictions)
            summaries = tf.summary.merge_all()

        ops = predictions
        ops.update(evals)
        ops["loss"] = loss
        ops["summaries"] = summaries
        ops["train_op"] = train_op

        return ops, graph

    def evaluate_graph(self):
        graph = tf.Graph()

        with graph.as_default() as g:
            predictions = self._infer()
            loss = self._calculate_loss(predictions["logits"])
            evals = self._evaluate_op(predictions)
            if self.quantization:
                tf.contrib.quantize.create_eval_graph()
            summaries = tf.summary.merge_all()

        ops = predictions
        ops.update(evals)
        ops["loss"] = loss
        ops["summaries"] = summaries

        return ops, graph

    def predict_graph(self):
        graph = tf.Graph()

        with graph.as_default() as g:
            predictions = self._infer()
            if self.quantization:
                tf.contrib.quantize.create_eval_graph()

        return predictions, graph

    def _set_drop_prob_to_LD(self, input_LD, drop_prob):
        """Adder/Updater of the key 'drop_prob:0' in a list of dictionaries

         Args:
            input_LD(list of dict): list of dictionaries
            drop_prob(float): drop probability

         Returns:
            The same list of dictionaries with a new(/updated) key 'drop_prob:0'
            with value drop_prob in each dictionary.

        """
        mode_d = {"drop_prob:0": drop_prob}

        for d in input_LD:
            d.update(mode_d)

        return input_LD

    def _init_dict_split_max(self, inputs, input_names):
        return init_dict_split_max(inputs, input_names)

    def _split_and_batch(self, inputs, input_names, batch_size,
                         validation_split, drop_prob):

        train_LD, val_LD = split_and_batch(inputs, input_names, batch_size,
                                           validation_split)

        if drop_prob is not None:
            train_LD = self._set_drop_prob_to_LD(train_LD, drop_prob)
            val_LD = self._set_drop_prob_to_LD(val_LD, 0.0)

        return train_LD, val_LD

    def fit(self,
            inputs,
            input_names=["features", "labels"],
            batch_size=1,
            validation_split=0,
            epochs=1,
            verbosity=0,
            drop_prob=None):
        """Train the model with given data and options.

        Args:
            inputs(list of np.ndarray): Data as list of arrays, one of which
                must be the labels.
            input_names(list of str): Names of inputs as used in the 
                placeholders of the model's functions.
            batch_size(int): Batch size with which the model will be trained.
            validation_split(float): Percentage of the data that is used for 
                validation (as such 0 <= validation_split < 1).
            epochs(int): The number of epochs to train (epochs >= 1).
            verbosity(int): If 0 will log only in the returned history. 
                If 1 tensorboard summaries will be written. If 2 result of ops
                will be printed for every batch (slow).
            drop_prob(float): Eventual drop_prob for dropout

        Returns:
            The history of the training: Dictionary of the result of every of 
            operation run on the graph during evaluation. 
            It's a dict with as jeys the name of the ops and as values the lists
            of the resulting value, one element in a list for each epoch.

        """
        ops, graph = self.train_ops_graph
        history = []

        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            train_LD, val_LD = self._split_and_batch(
                inputs, input_names, batch_size, validation_split, drop_prob)

            if verbosity >= 1:
                summary_writer_train = tf.summary.FileWriter(
                    self.tb_path_train, sess.graph)
                summary_writer_validation = tf.summary.FileWriter(
                    self.tb_path_val, sess.graph)

                run_metadata = tf.RunMetadata()
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                print(
                    "For training: tensorboard --logdir=" + self.tb_path_train)
                print(
                    "For validation: tensorboard --logdir=" + self.tb_path_val)
                i = 1
            else:
                run_metadata = None
                run_options = None

            for e in range(1, epochs + 1):
                sess.run(tf.local_variables_initializer())

                for train_dict in train_LD:
                    if verbosity >= 1:
                        if i % 20 == 0:
                            run_metadata = tf.RunMetadata()
                            run_options = tf.RunOptions(
                                trace_level=tf.RunOptions.FULL_TRACE)

                    out = sess.run(
                        ops,
                        feed_dict=train_dict,
                        options=run_options,
                        run_metadata=run_metadata)

                    if verbosity >= 1:
                        if i % 20 == 0:
                            summary_writer_train.add_run_metadata(
                                run_metadata, "step" + str(i))
                            run_metadata = None
                            run_options = None

                        summary_writer_train.add_summary(out["summaries"], i)
                        i = i + 1

                        if verbosity == 2:
                            print({
                                x: out[x]
                                for x in out if x in ["accuracy", "loss"]
                            })

                sess.run(tf.local_variables_initializer())

                for val_dict in val_LD:

                    if verbosity >= 1:
                        run_metadata = tf.RunMetadata()

                    out = sess.run(
                        {x: ops[x]
                         for x in ops if x not in ["train_op"]},
                        feed_dict=val_dict,
                        options=run_options,
                        run_metadata=run_metadata)

                    if verbosity >= 1:
                        summary_writer_validation.add_summary(
                            out["summaries"], e)
                        summary_writer_train.flush()
                        summary_writer_validation.flush()

                history.append(
                    {x: out[x]
                     for x in out if x in ["accuracy", "loss"]})

            if verbosity >= 1:
                summary_writer_train.close()
                summary_writer_validation.close()

            saver.save(sess, self.save_path.as_posix())

        return dict(
            zip(history[0],
                zip(*[out.values() for out in history])))  # Dict of lists

    def predict(self, inputs, input_names=["features"]):
        """Make predictions on the given data, by a trained model.

        Args:
            inputs(list of np.ndarray): Data as list of arrays.
            input_names(list of str): Names of inputs as used in the 
                placeholders of the model's functions.

        Returns:
            The predictions as dictionary composed of: logits, classes and 
            probabilities as np.ndarray (zero-axis indices inputs).

        """

        ops, graph = self.predict_ops_graph

        with tf.Session(graph=graph) as sess:
            input_LD = self._init_dict_split_max(inputs, input_names)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.save_path.as_posix())

            sess.run(tf.local_variables_initializer())
            out = []

            for input_dict in input_LD:
                out.append(sess.run(ops, feed_dict=input_dict))

        return out

    def evaluate(self, inputs, input_names=["features", "labels"]):
        """Evaluate trained model on the given data.

        Args:
            inputs(list of np.ndarray): Data as list of arrays, one of which
                must be the labels.
            input_names(list of str): Names of inputs as used in the 
                placeholders of the model's functions.

        Returns:
            The evaluation of the model as dictionary composed of 
            logits, classes, probabilities, loss and all metrics specified in
            the eval_fn specified at construction time as np.ndarray 
            (zero-axis indices inputs).

        """

        ops, graph = self.eval_ops_graph

        with tf.Session(graph=graph) as sess:
            input_LD = self._init_dict_split_max(inputs, input_names)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.save_path.as_posix())

            sess.run(tf.local_variables_initializer())
            out = []

            for input_dict in input_LD:
                out.append(sess.run(ops, feed_dict=input_dict))

        return out

    def load_model(self):
        """Load previously frozen model as graph with input and output names.
        
        Returns:
            The graph of the frozel model as well as input and output names for
            further use
            
        Under the hood uses general functions, just a wrapper with model 
        embedded info such as names and path.
        """
        return load_frozen_graph(
            (MODELS_DIR / self.name / 'model.pb').as_posix())

    def freeze(self, output_names=['softmax']):
        graph = self.predict_ops_graph[1]

        return freeze_graph(graph, output_names, self.save_path.as_posix())

    def save_frozen_graph(self):
        """Freeze the prediction graph with last saved data and write it to disk.

        The file will be written to the model dir as "self.name + '.pb'",
        in ProtoBuff format.
        """
        write_graph(self.freeze(), self.name + '.pb',
                    self.save_path.parent.as_posix())

    def save_optimazed_graph(self):
        """Optimize the prediction graph with last saved data and write it to disk.

        The file will be written to the model dir as "self.name + '.pb'",
        in ProtoBuff format.
        """
        write_graph(self.optimize(), self.name + '.pb',
                    self.save_path.parent.as_posix())

    def optimize(self,
                 add_transf=[],
                 input_names=['features'],
                 output_names=['softmax']):
        """Optimize the model graph for inference, quantize if constructed for it.

        Args:
            add_transf: Additional transformation to apply.
            

        Returns:
            The transformed optimized graph. If quantization was enabled at
            construction time then it will be finalized (from fake to real).
        """

        return optimize_for_inference(self.freeze(), input_names,
                                      output_names, self.quantization,
                                      add_transf)
