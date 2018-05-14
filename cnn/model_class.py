"""Module implementig TfClassifier class."""

import numpy as np
import tensorflow as tf
from pathlib import Path

from cnn.utils.save_models import MODELS_DIR


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
                 fake_quantization=False):
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
                and :method:`freeze` will implement a real quantization
                transformation on the graph.
        """

        self.name = name
        self.forward_pass_fn = forward_pass_fn
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.optimizer = optimizer
        self.fake_quantization = fake_quantization
        self.train_ops_graph = self.train_graph()
        self.eval_ops_graph = self.evaluate_graph()
        self.predict_ops_graph = self.predict_graph()
        self.tb_path = (Path("/tmp/log-tb/") / self.name).as_posix()
        self.save_path = (MODELS_DIR / (self.name + ".ckpt")).as_posix()

    def _infer(self):

        training_placeholder = tf.placeholder(tf.bool, shape=(), 
                                                    name='train_mode')

        logits = self.forward_pass_fn(
            train_mode_placeholder=training_placeholder)

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
            predictions = self._infer()
            loss = self._calculate_loss(predictions["logits"])
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

        return predictions, graph

    def _split_and_batch(self, inputs, input_names, batch_size, validation_split):

        n_samples = inputs[0].shape[0]
        input_tensors = list(
            map(lambda n: tf.get_default_graph().get_tensor_by_name(n + ':0'),
                input_names))
        input_DL = dict(zip(input_tensors, inputs))  # Dict from pairs of list

        for k, v in input_DL.items():
            input_DL[k] = np.split(v,
                                   [round((1 - validation_split) * n_samples)])

        input_LD = [dict(zip(input_DL, t))
                    for t in zip(*input_DL.values())]  #List of Dicts

        train_dict = input_LD[0]
        val_dict = input_LD[1]

        n_train_samples = train_dict[input_tensors[0]].shape[0]
        n_batches, drop = np.divmod(n_train_samples, batch_size)

        for k, v in train_dict.items():
            train_dict[k] = np.array_split(v, n_batches)

        train_LD = [
            dict(zip(train_dict, t)) for t in zip(*train_dict.values())
        ]

        val_dict.update({"train_mode:0": False})
        mode_t = {"train_mode:0": True}
        for d in train_LD:
            d.update(mode_t)

        return train_LD, val_dict

    def fit(self,
            inputs,
            input_names=["features", "labels"],
            batch_size=1,
            validation_split=0,
            epochs=1,
            verbosity=0):
        """Train the model with given data and optiions.
        
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
                If 1 tensorboard summaries will be written.
                If 2 result of ops will be printed for every batch (slow).

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

            train_LD, val_dict = self._split_and_batch(
                inputs, input_names, batch_size, validation_split)

            if verbosity >= 1:
                summary_writer = tf.summary.FileWriter(self.tb_path,
                                                       sess.graph)
                print("Launch: tensorboard --logdir=" + self.tb_path)
                i = 1

            for e in range(1, epochs + 1):
                sess.run(tf.local_variables_initializer())

                for train_dict in train_LD:
                    out = sess.run(ops, feed_dict=train_dict)
                    if verbosity >= 1:
                        summary_writer.add_summary(out["summaries"], i)
                        summary_writer.flush()
                        i = i + 1
                    if verbosity == 2:
                        print({x: out[x]
                             for x in out if x in ["accuracy", "mse", "loss"]})

                sess.run(tf.local_variables_initializer())
                out = sess.run(
                    {x: ops[x]
                     for x in ops if x not in ["train_op"]},
                    feed_dict=val_dict)

                history.append(out)
            
            if verbosity >=1:
                summary_writer.close()
            
            saver.save(sess, self.save_path)

        return dict(
            zip(history[0],
                zip(*[out.values() for out in history])))  #Dict of lists

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
        saver = tf.train.Saver()

        input_tensors = map(tf.get_default_graph().get_tensor_by_name,
                            input_names)
        input_dict = dict(zip(input_tensors, inputs))
        input_dict.update({self._training_placeholder: False})

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.save_path)

            out = sess.run(ops, feed_dict=input_dict)

        return out

    def evaluate(self, inputs, input_name=["features", "labels"]):
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
        saver = tf.train.Saver()

        input_tensors = map(tf.get_default_graph().get_tensor_by_name,
                            input_names)
        input_dict = dict(zip(input_tensors, inputs))
        input_dict.update({self._training_placeholder: False})

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.save_path)

            out = sess.run(ops, feed_dict=input_dict)

        return out

    def freeze(self, path):
        """Freezes the prediction graph and writes it to disk.
        
        Args:
            path(str): Absolute string representing path to dir.
            
        Returns:
            Absolute path to the written file.

        Variables becomes constants, eventually performs quantization if 
        enabled at construction time.
        The file will be written to the given path as "self.name + '.pb'",
        in ProtoBuff format.
        """
        pass

    def _transform():
        """Apply transformations to a graph"""
        pass
