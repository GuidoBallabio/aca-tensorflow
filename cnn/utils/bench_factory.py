from pathlib import Path

import perf

from cnn.utils.dataset import dataset_preprocessing_by_keras, load_cifar10
from cnn.utils.graph_manipulation import just_run_graph, load_frozen_graph
from cnn.utils.prep_inputs import init_dict_split_max, split_and_batch


class BenchmarkFactory:
    def __init__(self, frozen_graph_path, runner, name=None):
        if name is None:
            self.name = Path(frozen_graph_path).name[:-3]
        else:
            self.name = name

        self.runner = runner

        data = load_cifar10()
        x = dataset_preprocessing_by_keras(data[2])
        input_names = ["features"]
        output_names = ["softmax"]

        self.x_max_size_LD = init_dict_split_max([x], input_names)
        self.x_unit_size_LD = split_and_batch([x], input_names, 1, 0)[0]
        self.graph = load_frozen_graph(frozen_graph_path)
        self.predict_test = lambda: just_run_graph(self.graph, self.x_max_size_LD, output_names)
        self.predict_single = lambda: just_run_graph(self.graph, self.x_unit_size_LD, output_names)

    def bench(self):
        self.runner.bench_func('max_batch', self.predict_test)
        self.runner.bench_func('single_sample', self.predict_test)
