# CLAUDE.md

**Project**: Multi-label audio tagging with noisy labels (PyTorch)
**Dataset**: FSDKaggle2019 (https://zenodo.org/records/3612637)
**Author**: Acoustic engineer + ML background
**Goal**: Portfolio piece demonstrating audio ML + signal processing expertise

## Task Execution

**IMPORTANT**: Before starting any task:
1. Read `.blueprints/TASK_QUEUE.md` to see current task
2. Read the specific task file (`.blueprints/tasks/XXX_task_name.md`)
3. Follow the SPEC section exactly
4. Run tests: `uv run pytest -x`
5. Mark task DONE in `TASK_QUEUE.md` when complete

For architectural decisions, reference `.blueprints/decisions/DXXX_*.md` files.

## Common Commands

```bash
# Run tests (do this after every task!)
uv run pytest -x                  # Stop on first failure
uv run pytest tests/test_file.py  # Specific test

# Code quality
uv run ruff check --fix .
uv run ruff format .

# Training
make train      # Baseline
make train-cnn  # CNN model
```

## Project Structure

```
src/
├── core/          # device.py, config.py, utils.py
├── data/          # dataset.py, loader.py, augmentation.py
├── features/      # signal_tools.py (SuperGaussian filters), spectrogram_optimized.py
├── models/        # baseline_cnn.py, efficient_cnn.py, components.py
├── training/      # losses.py, metrics.py, trainer.py, callbacks.py
├── visualization/ # plots.py
└── scripts/       # train.py, evaluate.py, devel.py

.blueprints/       # Task definitions and decisions (READ THESE!)
├── OVERVIEW.md    # Project architecture summary
├── TASK_QUEUE.md  # Current work queue with status
├── tasks/         # Individual task specifications
└── decisions/     # Architecture decision records
```

**Key Features**:
- SuperGaussian filter banks (analytical, efficient frequency decomposition)
- Psychoacoustic normalization (20 dB temporal masking + 60 dB global floor)
- MPS acceleration for M4 Apple Silicon
- See `.blueprints/OVERVIEW.md` for detailed architecture

## Code Quality Standards

**Type Hints**: All functions must include complete type hints for parameters and return values.

**Dataclasses**: Use `@dataclass` for structured data to improve code clarity and maintainability.

**Computational Optimization**:
- Pre-compute scalar operations outside vector operations
- Example: `(2 * pi * sigma) * vector` is preferred over `2 * pi * sigma * vector`
- This makes vector operations explicit and reduces unnecessary operations
- Leverage NumPy broadcasting effectively
- Use analytical solutions over numerical when possible (as in signal_tools.py)
- **Avoid two-step array creation**: Create arrays directly with `linspace` or similar
  - ❌ Bad: `bin_rel = np.arange(-n, n+1); freq = bin_rel * df`
  - ✅ Good: `freq = np.linspace(-n*df, n*df, 2*n+1)`
- **Compute indices analytically when possible**: Avoid boolean masking when indices follow a known pattern
  - ❌ Bad: `mask = (bins >= 0) & (bins < max); valid = bins[mask]`
  - ✅ Good: `i_start = max(0, ...); i_end = min(max, ...); valid = bins[i_start:i_end]`
  - Use boolean masking only when the condition is truly data-dependent and unpredictable

**Code Readability**:
- Human-readable variable names that reflect their physical/mathematical meaning
- Clear function documentation with parameter descriptions
- Logical code organization with single responsibility principle

**Code Organization & Reusability**:
- Group coherent functionality in dedicated files (e.g., all device/GPU logic in `src/device.py`)
- Extract complex logic into well-named functions
- Keep main files clean and high-level - details belong in utility modules
- Avoid over-abstraction - prefer simple, direct solutions over complex hierarchies
- For plotting: use matplotlib OOP API (fig, ax objects), but wrap plot-related code in functions
- Pack related utilities together (device management, feature extraction, etc.)

**Avoiding Deprecated APIs**:
- Always use current PyTorch APIs (e.g., `torch.amp.autocast('mps', ...)` not `torch.cuda.amp.autocast(...)`)
- Check documentation for deprecation warnings before using library functions
- Use modern library patterns and idioms

**Performance Considerations**:
- Utilize GPU acceleration via PyTorch when applicable
- Optimize for Apple Silicon (M4) using MPS (Metal Performance Shaders) backend
- Profile computationally intensive operations
- Cache expensive computations when appropriate

## Hardware & Acceleration

**Target Platform**: MacBook Air M4

**GPU Acceleration**:
- Use PyTorch's MPS backend for GPU acceleration on Apple Silicon
- Check device availability: `torch.backends.mps.is_available()`
- Move tensors and models to MPS device: `.to('mps')`
- Be aware of MPS-specific limitations and fallback to CPU when necessary

**Best Practices**:
- Default to MPS for training and inference when available
- Keep data loading on CPU, move batches to MPS during training
- Use `torch.set_default_device('mps')` for automatic device placement
- Monitor memory usage on M4 (unified memory architecture)

## Important Notes

- All module execution uses `uv run python -m module_name` pattern
- Configuration files must specify absolute paths for `data.base_dir`
- The project is configured for 16kHz sample rate and 5-second clips by default
- Model outputs, metrics, and figures are saved to `reports/` subdirectories
- Custom Claude Code configurations are stored in `.claude/` directory (git-ignored)