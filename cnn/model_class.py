"""Module implementig TfClassifier class."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

from cnn.utils.save_models import MODELS_DIR, load_frozen_graph, write_graph, transform_graph

HALF_MAX_BATCH_SIZE = 2000


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
                and :method:`freeze` will enable to implement a real 
                quantization through the command line tool GraphTransform.
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
        self.tb_path_train = (
            Path("/tmp/log-tb/") / self.name / "training").as_posix()
        self.tb_path_val = (
            Path("/tmp/log-tb/") / self.name / "validation").as_posix()
        self.save_path = MODELS_DIR / self.name / "model.ckpt"

    def _infer(self, train_mode=False):

        self.drop_prob_placeholder = tf.placeholder_with_default(
            0.0, (), name="drop_prob")

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

    def _init_dict(self, inputs, input_names):
        """
        Example:
            inputs = [1,1]
            input_names = ['labels','drop_prob']
        Outputs: 
            input_tensors = [<labels' tensor>, <drop_prob's tensor>]
            input_DL = {<labels' tensor> : 1, <drop_prob's tensor> : 1}
        """
        input_tensors = [
            tf.get_default_graph().get_tensor_by_name(n + ':0')
            for n in input_names
        ]
        input_DL = dict(zip(input_tensors, inputs))  # Dict from pairs of list

        return input_tensors, input_DL

    def _split_data_dict_in_perc(self, input_dict, n_samples, percs):
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

        input_LD = [
            dict(zip(input_dict, t)) for t in zip(*input_dict.values())
        ]  # List of Dicts

        return input_LD

    def _batch_data_dict(self, input_dict, n_samples, batch_size):
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
        """
        Example:
            inputs = [4200 features 2x2x2, 4200 labels]
            input_names = ['features','labels']
        Output: [{<features' tensor> : 2200 features 2x2x2, <labels' tensor> :
            2200 labels 2x2x2},
            {<features' tensor> : 2000 features 2x2x2, <labels' tensor> : 
            2000 labels 2x2x2}]
        """
        input_dict = self._init_dict(inputs, input_names)[1]
        n_samples = inputs[0].shape[0]

        out_LD = self._batch_data_dict(input_dict, n_samples,
                                       HALF_MAX_BATCH_SIZE)

        return out_LD

    def _split_and_batch(self, inputs, input_names, batch_size,
                         validation_split, drop_prob):
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

        input_tensors, input_DL = self._init_dict(inputs, input_names)

        input_LD = self._split_data_dict_in_perc(input_DL, n_samples,
                                                 np.array(
                                                     [1 - validation_split]))

        train_dict = input_LD[0]
        val_dict = input_LD[1]

        n_train_samples = train_dict[input_tensors[0]].shape[0]

        train_LD = self._batch_data_dict(train_dict, n_train_samples,
                                         batch_size)

        val_LD = self._batch_data_dict(val_dict, n_samples - n_train_samples,
                                       HALF_MAX_BATCH_SIZE)

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

    def _freeze_graph(self, graph=None, output_names=['softmax']):
        if graph is None:
            graph = self.predict_ops_graph[1]

        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.save_path.as_posix())

            constant_graph_def = graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(), output_names)

        return constant_graph_def

    def save_frozen_graph(self):
        """Freeze the prediction graph with last saved data and write it to disk.

        Args:
            graph: Graph to freeze, defaults to self.predict_graph
            output_names: List of strings. Names of output op

        The file will be written to the model dir as "self.name + '.pb'",
        in ProtoBuff format.
        """
        write_graph(self._freeze_graph(), 'model.pb',
                    self.save_path.parent.as_posix())

    def optimize_for_inference(self,
                               add_transf=[],
                               input_names=['features'],
                               output_names=['softmax'],
                               graph_def=None):
        """Optimize the model graph for inference, quantize if constructed for it.

        Args:
            add_transf: Additional transformation to apply.
            

        Returns:
            The transformed optimized graph. If quantization was enabled at
            construction time then it will be finalized (from fake to real).
        """
        transforms = [
            "strip_unused_nodes(type=float)",
            "remove_nodes(op=Identity, op=CheckNumerics)",
            "fold_constants(ignore_errors=true)", "fold_batch_norms",
            "fold_old_batch_norms", "sort_by_execution_order"
        ]

        if self.fake_quantization:
            transforms = ["add_default_attributes"] + transforms[:-1] + [
                "quantize_weights", "quantize_nodes", "sort_by_execution_order"
            ]

        for transf in add_transf:
            if transf not in transforms:
                transforms.append(transf)

        if graph_def is None:
            graph_def = self._freeze_graph()

        return transform_graph(graph_def, input_names, output_names,
                               transforms)
