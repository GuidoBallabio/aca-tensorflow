nets = dense
net_names = $(addsuffix _cnn, ${nets})
opt_nets = $(addsuffix _opt, ${net_names})

py_files = $(addprefix cnn/, $(addsuffix .py, $(nets)))
trainings = $(addprefix models/, $(addsuffix .pb, $(opt_nets)))
benchmarks = $(addprefix results/, $(addsuffix .json, $(opt_nets)))

BENCH_OPT = "--verbose" # --rigorous -l LOOPS/--loops=LOOPS -w WARMUPS/--warmups=WARMUPS
						# --min-time=MIN_TIME
TRAIN_OPT = "-v"

$(benchmarks): results/%.json: models/%.pb
	python run_bench.py $(basename $(notdir $<)) -o "$@" $(BENCH_OPT)

$(trainings): models/%_cnn_opt.pb: cnn/%.py
	python -m cnn.$(basename $(notdir $<)) "$(basename $(notdir $<))_cnn" $(TRAIN_OPT)

$(py_files): ;

all: clean init train benchmark

init:
	pip install -U -r requirements.txt
	python -m cnn.utils.dataset

beautify:
	yapf -i --recursive .
	isort -ac --recursive .

test:
	python -m pytest

train: $(trainings)

benchmark: $(benchmarks)

clean:
	rm cnn/data/*
	rm cnn/models/*
	rm /tmp/log-tb/*
	rm results/*

.PHONY: all init beautify test clean
