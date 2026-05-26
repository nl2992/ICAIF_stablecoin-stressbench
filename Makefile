.PHONY: install test lint format run-live pull-archive build-features train evaluate pipeline clean

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
	python scripts/pull_data.py

build-features:
	python scripts/build_features.py

train:
	python scripts/train_models.py

evaluate:
	python scripts/evaluate_models.py

pipeline:
	python scripts/run_pipeline.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
