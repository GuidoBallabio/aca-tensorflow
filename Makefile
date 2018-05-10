init:
	pip install -r requirements.txt
	python -m cnn.utils

fit:
	python -m cnn.dense

clean:
	rm -r cnn/data

.PHONY: init fit clean
