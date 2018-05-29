nets = dense
opt_nets = $(addsuffix _opt, ${nets})
quant_nets = $(addsuffix _quant, ${nets})
all_nets = $(opt_nets) $(quant_nets)

py_files = $(addprefix cnn/, $(addsuffix .py, $(nets)))
train_opt = $(addprefix models/, $(addsuffix .pb, $(opt_nets)))
train_quant = $(addprefix models/, $(addsuffix .pb, $(quant_nets)))
benchmarks = $(addprefix results/, $(addsuffix .json, $(all_nets)))

BENCH_OPT = "--verbose" # --rigorous -l LOOPS/--loops=LOOPS -w WARMUPS/--warmups=WARMUPS
						# --min-time=MIN_TIME
TRAIN_OPT = "-v"
QUANTIZE = "--quantization"

$(benchmarks): results/%.json: models/%.pb
	python run_bench.py $(basename $(notdir $<)) -o "$@" $(BENCH_OPT)

$(train_opt): models/%_opt.pb: cnn/%.py
	python -m cnn.$(basename $(notdir $<)) "$(basename $(notdir $@))" $(TRAIN_OPT)

$(train_quant): models/%_quant.pb: cnn/%.py
	python -m cnn.$(basename $(notdir $<)) "$(basename $(notdir $@))" $(QUANTIZE) $(TRAIN_OPT)

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

train: $(train_opt) $(train_quant)

benchmark: $(benchmarks)

clean:
	rm cnn/data/*
	rm cnn/models/*
	rm /tmp/log-tb/*
	rm results/*

.PHONY: all init beautify test clean
