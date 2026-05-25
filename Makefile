.PHONY: install test lint run-live pull-archive build-silver features labels train evaluate clean

install:
	pip install -e .

test:
	pytest tests/

lint:
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

run-live:
	python scripts/start_live_capture.py

pull-archive:
	python scripts/pull_binance_archive.py

build-silver:
	python scripts/build_silver.py

features:
	python scripts/build_features.py

labels:
	python scripts/build_labels.py

train:
	python scripts/train_baselines.py

evaluate:
	python scripts/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
