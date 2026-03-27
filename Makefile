dev:
	uv run python -m src.devel --config configs/baseline.yaml

train:
	uv run python -m src.train --config configs/baseline.yaml

train-cnn:
	uv run python -m src.train --config configs/cnn.yaml

evaluate:
	uv run python -m src.evaluate --config configs/cnn.yaml


check:
	uv run ruff check .

format:
	uv run ruff format .

