all: init dense benchmarks

init:
	pip install -U -r requirements.txt
	python -m cnn.utils

dense:
	python -m cnn.dense

conv:


complex:


benchmark:
	jupyter notebook Benchmarks.ipynb

clean:
	rm -r cnn/data
	rm cnn/models/*

.PHONY: init dense benchmarks clean
