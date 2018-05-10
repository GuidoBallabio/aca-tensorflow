init:
	pip install -r requirements.txt
	python -m cnn.utils

dense:
	python -m cnn.dense

conv:

complex:

clean:
	rm -r cnn/data
	rm -r cnn/models

.PHONY: init dense clean
