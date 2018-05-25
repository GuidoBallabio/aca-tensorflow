import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

from cnn.utils.prep_inputs import init_dict_split_max

MODELS_DIR = Path(__file__).parent.parent / 'models'


def transform_graph(graph_def, input_names, output_names, transforms):
    out_graph_def = TransformGraph(graph_def, input_names, output_names,
                                   transforms)

    out_graph = tf.Graph()
    with out_graph.as_default():
        tf.import_graph_def(out_graph_def, name='')

    return out_graph


def write_graph(graph_def, name, dir_path=MODELS_DIR.as_posix()):
    tf.train.write_graph(graph_def, dir_path, name, as_text=False)


def freeze_graph(graph, output_names, save_path):
    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path)

        constant_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), output_names)

    return constant_graph_def


def load_frozen_graph(frozen_graph_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    # Then, we import the graph_def into a new Graph and returns it
    with graph.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')

    return graph


def predict_from_frozen(graph, inputs, input_names, output_names):

    input_LD = init_dict_split_max(inputs, input_names)

    with tf.Session(graph=graph) as sess:
        out = []
        for input_dict in input_LD:
            out.append(sess.run(ops, feed_dict=input_dict))

    return out


def optimize_for_inference(graph_def,
                           input_names,
                           output_name,
                           quantization,
                           add_transf=[]):
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

    if quantization:
        transforms = ["add_default_attributes"] + transforms[:-1] + [
            "quantize_weights", "quantize_nodes", "strip_unused_nodes",
            "sort_by_execution_order"
        ]

    for transf in add_transf:
        if transf not in transforms:
            transforms.append(transf)

    return transform_graph(graph_def, input_names, output_names, transforms)
