from pathlib import Path

import numpy as np
import perf

from cnn.utils.dataset import dataset_preprocessing_by_keras, load_cifar10
from cnn.utils.graph_manipulation import (just_run_graph, load_frozen_graph,
                                          run_graph_and_analyze)
from cnn.utils.prep_inputs import init_dict_split_max, split_and_batch

PERC_DATA_KEEP = 0.1
SOFT_ANALYSIS = True

class BenchmarkFactory:
    def __init__(self, frozen_graph_path, runner, name=None, analysis=False):
        if name is None:
            self.name = Path(frozen_graph_path).name[:-3]
        else:
            self.name = name

        self.runner = runner
        self.analysis = analysis

        #Input preparation
        data = load_cifar10()
        x = dataset_preprocessing_by_keras(data[2])
        x = np.split(x, 1/PERC_DATA_KEEP)[0]
        input_names = ["features"]
        output_names = ["softmax:0"]

        self.x_1_size_LD = split_and_batch([x], input_names, 1, 0)[0]
        self.x_16_size_LD = split_and_batch([x], input_names, 16, 0)[0]
        self.x_32_size_LD = split_and_batch([x], input_names, 32, 0)[0]
        self.x_64_size_LD = split_and_batch([x], input_names, 64, 0)[0]
        self.x_max_size_LD = init_dict_split_max([x], input_names)
        self.graph = load_frozen_graph(frozen_graph_path)

        #Just run
        self.predict_1 = lambda: just_run_graph(self.graph, self.x_1_size_LD, output_names)
        self.predict_16 = lambda: just_run_graph(self.graph, self.x_16_size_LD, output_names)
        self.predict_32 = lambda: just_run_graph(self.graph, self.x_32_size_LD, output_names)
        self.predict_64 = lambda: just_run_graph(self.graph, self.x_64_size_LD, output_names)
        self.predict_max = lambda: just_run_graph(self.graph, self.x_max_size_LD, output_names)

        #Analysis
        self.predict_1_analysis = lambda: run_graph_and_analyze(self.graph, self.x_1_size_LD, output_names)
        self.predict_16_analysis = lambda: run_graph_and_analyze(self.graph, self.x_16_size_LD, output_names)
        self.predict_32_analysis = lambda: run_graph_and_analyze(self.graph, self.x_32_size_LD, output_names)
        self.predict_64_analysis = lambda: run_graph_and_analyze(self.graph, self.x_64_size_LD, output_names)
        self.predict_max_analysis = lambda: run_graph_and_analyze(self.graph, self.x_max_size_LD, output_names)

    def bench(self):
        self.analysis = True
        if self.analysis and not SOFT_ANALYSIS:
            self.runner.bench_func('1-batch_analysis', self.predict_1_analysis)
            self.runner.bench_func('16-batch_analysis',
                                   self.predict_16_analysis)
            self.runner.bench_func('32-batch_analysis',
                                   self.predict_32_analysis)
            self.runner.bench_func('64-batch_analysis',
                                   self.predict_64_analysis)
            self.runner.bench_func('max-batch_analysis',
                                   self.predict_max_analysis)
        elif not self.analysis and not SOFT_ANALYSIS:
            self.runner.bench_func('1-batch', self.predict_1)
            self.runner.bench_func('16-batch', self.predict_16)
            self.runner.bench_func('32-batch', self.predict_32)
            self.runner.bench_func('64-batch', self.predict_64)
            self.runner.bench_func('max-batch', self.predict_max)
        elif self.analysis and SOFT_ANALYSIS:
            self.runner.bench_func('1-batch_analysis',
                                   self.predict_1_analysis)
        else:
            self.runner.bench_func('1-batch', self.predict_1)

