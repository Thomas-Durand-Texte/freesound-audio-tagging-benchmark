dev-benchmark:
	# Benchmark all methods
	uv run python -m src.devel --config configs/your_config.yaml --test benchmark --n-samples 1

	# Include GPU testing
	#python -m src.devel --config configs/baseline.yaml --test benchmark --n-samples 1 --test-gpu

dev-spectrogram:
	# Test spectrogram differences
	uv run python -m src.devel --config configs/baseline.yaml --test spectrogram --n-samples 1

dev-envelope:
	# Test signal processing (envelope patterns and filter banks)
	uv run python -m src.devel --config configs/baseline.yaml --test signal

	# Test audio dataset loading
	#uv run python -m src.devel --config configs/your_config.yaml --test audio --n-samples 5


explore-compute:
	# Compute statistics (detects and warns about problematic files)
	uv run python -m src.explore --config configs/baseline.yaml --compute-stats

explore-spectrograms:
	# Generate spectrograms (skipping problematic files by default)
	uv run python -m src.explore --config configs/baseline.yaml --compute-spectrograms --max-per-label 5

explore-display:
	# Display statistics
	uv run python -m src.explore --config configs/baseline.yaml --display-stats



train:
	uv run python -m src.train --config configs/baseline.yaml

evaluate:
	uv run python -m src.evaluate --config configs/baseline.yaml


check:
	uv run ruff check .

format:
	uv run ruff format .

