from __future__ import absolute_import, division, print_function

import re
import time as tm
from pathlib import Path

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.profiler import option_builder, tfprof_logger
from tensorflow.python.profiler.model_analyzer import Profiler
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


def run_graph_and_analyze(graph, input_LD, output_names, timeline=False):
    step_counter = 0
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').build()
    with tf.contrib.tfprof.ProfileContext(
            '/tmp/benchmark', trace_steps=[], dump_steps=[]) as pctx:
        with tf.Session(graph=graph) as sess:
            profiler = Profiler(sess.graph)
            run_meta = tf.RunMetadata()
            for input_dict in input_LD:
                if step_counter % 5 == 0:
                    pctx.trace_next_step()
                    pctx.dump_next_step()
                    sess.run(
                        output_names,
                        feed_dict=input_dict,
                        options=tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE),
                        run_metadata=run_meta)
                    pctx.profiler.profile_operations(options=opts)
                    profiler.add_step(step_counter, run_meta)
                    if not timeline:
                        #Time (option 1 on 2)
                        opts = option_builder.ProfileOptionBuilder.time_and_memory(
                        )
                        profiler.profile_operations(options=opts)
                    else:
                        #Timeline (option 2 on 2)
                        timestamp = int(tm.time())
                        filename = '/tmp/timeline' + str(timestamp) + '.json'
                        opts = (option_builder.ProfileOptionBuilder(
                            option_builder.ProfileOptionBuilder.
                            time_and_memory()).with_step(timestamp)
                                .with_timeline_output(filename).build())
                        profiler.profile_graph(options=opts)
                else:
                    sess.run(output_names, feed_dict=input_dict)
                step_counter += 1

def just_run_graph(graph, input_LD, output_names):
    with tf.Session(graph=graph) as sess:
        for input_dict in input_LD:
            sess.run(output_names, feed_dict=input_dict)


def predict_from_frozen(graph, inputs, input_names, output_names):

    input_LD = init_dict_split_max(inputs, input_names)

    with tf.Session(graph=graph) as sess:
        out = []
        for input_dict in input_LD:
            out.append(sess.run(output_names, feed_dict=input_dict))

    return out


def optimize_for_inference(graph_def,
                           input_names,
                           output_names,
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


def convert_graph_to_dot(input_graph, output_dot, is_input_graph_binary):
    graph = graph_pb2.GraphDef()
    with open(input_graph, "rb") as fh:
        if is_input_graph_binary:
            graph.ParseFromString(fh.read())
        else:
            text_format.Merge(fh.read(), graph)
    with open(output_dot, "wt") as fh:
        print("digraph graphname {", file=fh)
        for node in graph.node:
            output_name = node.name
            print(
                "  \"" + output_name + "\" [label=\"" + node.op + "\"];",
                file=fh)
            for input_full_name in node.input:
                parts = input_full_name.split(":")
                input_name = re.sub(r"^\^", "", parts[0])
                print(
                    "  \"" + input_name + "\" -> \"" + output_name + "\";",
                    file=fh)
        print("}", file=fh)
        print("Graph '%s' has been converted to DOT file: '%s'." %
              (input_graph, output_dot))
