all: init benchmarks

init:
	pip install -U -r requirements.txt
	python -m cnn.utils.dataset

dense:
	python -m cnn.dense

beautify:
	yapf -i --recursive .
	isort -ac --recursive .

test:
	python -m pytest

benchmark:
	jupyter notebook Benchmarks.ipynb

clean:
	rm cnn/data/*
	rm cnn/models/*
	rm /tmp/log-tb/*

.PHONY: init dense benchmarks clean
