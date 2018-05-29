#!/usr/bin/env python3
from pathlib import Path

import perf

from cnn.utils.bench_factory import BenchmarkFactory

GRAPHS_DIR = (Path(__file__).parent / "models/").absolute()
FROZEN_NETS = ["dense_opt", "dense_quant"]


def main(name, runner):
    bench_suite = BenchmarkFactory((GRAPHS_DIR / (name + ".pb")).as_posix(),
                                   runner)
    bench_suite.bench()


def add_cmdline_args(cmd, args):
    if args.net_name:
        cmd.append(args.net_name)


if __name__ == "__main__":
    runner = perf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument("net_name", nargs='?', choices=FROZEN_NETS)
    args = runner.parse_args()

    main(args.net_name, runner)
