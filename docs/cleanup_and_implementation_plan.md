# Codebase Analysis & Cleanup Plan
**Goal: Build and train a deep neural network for automatic audio labeling**

**Date**: 2026-03-29
**Project**: Freesound Audio Tagging Benchmark

---

## 📊 Current State Summary

### Implemented ✅
- **Signal Processing**: Complete filter bank implementation (Gaussian, SuperGaussian)
- **Data Pipeline**: DuckDB-based metadata management, audio loading, PyTorch Dataset
- **Feature Extraction**: Multiple optimized spectrogram methods (FFT, GPU, multi-resolution)
- **Exploration Tools**: Statistics computation, visualization, spectrogram generation
- **Configuration**: YAML-based configuration system
- **Code Quality**: Type hints, docstrings, ruff linting configured

### Missing ❌
- **ML Training Pipeline**: No training loop, model definitions, or trainer class
- **Model Architecture**: No CNN implementation despite config references
- **Loss Functions**: No multi-label classification loss
- **Evaluation**: No metrics (mAP, F1, precision, recall)
- **Data Augmentation**: No augmentation strategies
- **Inference**: No prediction/inference script
- **Testing**: No test suite (tests/ directory missing)

---

## 🔍 Issues Identified

### 1. Critical: Missing ML Components (BLOCKER)

**Problem**: Cannot train models without these core components

**Missing Files:**
```
src/train.py         # Referenced in Makefile but doesn't exist
src/evaluate.py      # Referenced in Makefile but doesn't exist
src/models.py        # No model architectures defined
src/losses.py        # No loss functions
src/metrics.py       # No evaluation metrics
src/augmentations.py # No data augmentation
```

**Impact**: **HIGH** - Core functionality missing for ML pipeline

**Config References**:
- `baseline.yaml` defines `model.channels`, `model.kernel_size` → No corresponding implementation
- `configs/cnn.yaml` exists but is **empty** (1 line)
- `features.type: log_mel_spectrogram` → Not used anywhere (code uses SuperGaussian)

---

### 2. Code Redundancy & Unused Components

#### 2a. Multiple Spectrogram Methods
**File**: `src/spectrogram_optimized.py` (849 lines)

**Contains 4+ methods:**
1. FFT-based (optimized)
2. Overlap-add convolution
3. STFT filter bank
4. GPU filter bank
5. MultiResolutionFilterBank

**Analysis**:
- `MultiResolutionFilterBank` is the **current choice** (used in `explore.py`, best performance)
- Other methods are for **benchmarking/comparison** (used in `devel.py` benchmarks)
- **STFTFilterBank** and **GPUFilterBank** classes (~200 lines) might not be actively used

**Recommendation**: **KEEP ALL** for now
- Useful for research and comparison
- Move to `src/benchmarks/` module later if needed
- Mark experimental methods clearly in docstrings

---

#### 2b. Trivial main.py
**File**: `main.py` (7 lines)

```python
def main():
    print("Hello from freesound-audio-tagging-benchmark!")
```

**Analysis**: Placeholder, never referenced in Makefile or docs

**Recommendation**: **DELETE** or repurpose as project entry point once training pipeline exists

---

#### 2c. Unused Development Scripts
**Files**:
- `src/devel.py` (378 lines) - Extensive testing/prototyping
- `src/spectrogram.py` (444 lines) - Comparison utilities with librosa

**Analysis**:
- `devel.py`: Used for development (audio playback, benchmarks) → **KEEP** for development workflows
- `spectrogram.py`: `compare_spectrograms()` used in `devel.py` → **KEEP** for validation

**Recommendation**: Move to `src/dev/` or `tools/` subdirectory to clearly separate from production code

---

### 3. Configuration Inconsistencies

#### 3a. Features vs Spectrogram Mismatch
**Config**: `baseline.yaml`

```yaml
features:
  type: log_mel_spectrogram  # ← NOT USED
  n_mels: 128
  n_fft: 1024
  hop_length: 256

spectrogram:  # ← ACTUALLY USED
  f_min: 20.0
  f_mid: 1000.0
  f_max: 8000.0
  n_bands: 128
```

**Impact**: Confusing, misleading configuration

**Recommendation**:
- **Option A**: Remove `features` section (SuperGaussian is the chosen approach)
- **Option B**: Add feature extraction abstraction to support both methods
- **Preference**: **Option A** (simplify) unless mel spectrogram baseline is needed for comparison

---

#### 3b. Empty cnn.yaml
**File**: `configs/cnn.yaml` (1 line, empty)

**Referenced in**: `Makefile` (train-cnn, evaluate targets)

**Recommendation**: **DELETE** or populate with actual CNN-specific config once models exist

---

#### 3c. Makefile Inconsistencies
**Issues**:
```makefile
# Some targets use 'python', some use 'uv run python'
explore-compute:
	python -m src.explore      # ← Inconsistent

dev-envelope:
	uv run python -m src.devel  # ← Consistent with uv
```

**Recommendation**: **Standardize** to `uv run python` everywhere (project uses `uv` as package manager)

---

### 4. Code Organization & Structure

#### 4a. Flat src/ Directory
**Current**:
```
src/
├── __init__.py
├── data.py
├── device.py
├── devel.py
├── explore.py
├── plot_utils.py
├── signal_tools.py
├── spectrogram.py
├── spectrogram_optimized.py
└── utils.py
```

**Recommendation**: **Refactor** into logical subdirectories (do this **AFTER** ML pipeline is complete):

```
src/
├── __init__.py
├── core/              # Core utilities
│   ├── device.py
│   ├── utils.py
│   └── config.py
├── data/              # Data handling
│   ├── dataset.py
│   ├── loader.py
│   └── augmentations.py
├── features/          # Feature extraction
│   ├── signal_tools.py
│   ├── spectrogram.py
│   └── spectrogram_optimized.py
├── models/            # Model architectures
│   ├── cnn.py
│   └── base.py
├── training/          # Training pipeline
│   ├── train.py
│   ├── trainer.py
│   ├── losses.py
│   └── metrics.py
├── evaluation/        # Evaluation
│   └── evaluate.py
├── visualization/     # Plotting
│   └── plot_utils.py
└── dev/               # Development tools
    ├── devel.py
    └── explore.py
```

**Impact**: **MEDIUM** - Better organization, but **DO LATER** (after core ML pipeline works)

---

### 5. Missing Documentation

**Issues**:
- No API documentation (consider Sphinx)
- `README.md` likely needs updating with current architecture
- No docstrings for some utility functions
- No usage examples in docs/

**Recommendation**: **LOW PRIORITY** - Add after ML pipeline is functional

---

### 6. Testing Infrastructure

**Current**: `tests/` directory doesn't exist (but referenced in `pyproject.toml`)

**Recommendation**: **MEDIUM PRIORITY**
- Create `tests/` directory
- Add unit tests for:
  - Data loading
  - Feature extraction
  - Model forward pass
  - Metrics computation
- Integration tests for training pipeline

---

## 📋 Prioritized Action Plan

### 🔴 CRITICAL - Must Do First (BLOCKERS)

#### 1. Implement ML Training Pipeline
**Priority**: **HIGHEST** - Core functionality

**Tasks**:
1. **Create `src/models.py`**:
   ```python
   class AudioCNN(nn.Module):
       """Multi-label audio classification CNN"""
       # Implement architecture from baseline.yaml config
       # Input: (batch, channels=1, freq=128, time=T)
       # Output: (batch, num_classes)
   ```

2. **Create `src/losses.py`**:
   ```python
   # Multi-label classification losses
   - BCEWithLogitsLoss (baseline)
   - Focal Loss (handle class imbalance)
   - ASL Loss (asymmetric loss for multi-label)
   ```

3. **Create `src/metrics.py`**:
   ```python
   # Multi-label metrics
   - mAP (mean Average Precision) - PRIMARY METRIC
   - F1 score (micro, macro, per-class)
   - Precision, Recall
   - AUC-ROC
   ```

4. **Create `src/train.py`**:
   ```python
   class Trainer:
       """Orchestrates training loop"""
       def __init__(self, model, train_loader, val_loader, config)
       def train_epoch()
       def validate()
       def save_checkpoint()
       def load_checkpoint()

   def main():
       """Entry point from Makefile"""
       # Load config
       # Initialize model, data loaders
       # Train loop
       # Save best model
   ```

5. **Create `src/evaluate.py`**:
   ```python
   def evaluate_model(model, test_loader, config):
       """Comprehensive evaluation on test set"""
       # Compute metrics
       # Generate confusion matrix
       # Per-class analysis
       # Save results to reports/
   ```

**Estimated Effort**: **2-3 days** (assuming standard CNN architecture)

---

#### 2. Data Augmentation
**Priority**: **HIGH** - Critical for model generalization

**Tasks**:
- Create `src/augmentations.py`:
  ```python
  # Audio augmentations
  - Time stretching
  - Pitch shifting
  - Time masking (SpecAugment)
  - Frequency masking
  - Gaussian noise
  - Mixup (mix two samples)
  ```

**Estimated Effort**: **1 day**

---

#### 3. Fix Config Inconsistencies
**Priority**: **HIGH** - Needed before training

**Tasks**:
1. Remove unused `features` section from `baseline.yaml`
2. Populate or delete `configs/cnn.yaml`
3. Standardize Makefile to use `uv run python` everywhere
4. Add `num_classes` to config (count from vocabulary)

**Estimated Effort**: **2 hours**

---

### 🟡 MEDIUM - Important but Not Urgent

#### 4. Code Organization Refactor
**Priority**: **MEDIUM** - Do **AFTER** ML pipeline works

**Tasks**:
- Restructure `src/` into subdirectories (see structure above)
- Update imports across all files
- Test that everything still works

**Estimated Effort**: **1 day**

**⚠️ WARNING**: Do this AFTER training pipeline is functional and tested

---

#### 5. Testing Infrastructure
**Priority**: **MEDIUM**

**Tasks**:
1. Create `tests/` directory structure:
   ```
   tests/
   ├── test_data.py
   ├── test_models.py
   ├── test_metrics.py
   └── test_training.py
   ```
2. Write unit tests for core components
3. Add integration tests for training

**Estimated Effort**: **2-3 days**

---

#### 6. Optimization Review
**Priority**: **MEDIUM** - After initial training works

**Tasks**:
- Profile training loop (identify bottlenecks)
- Optimize data loading (parallel workers, prefetch)
- Review GPU utilization (MPS backend)
- Consider mixed precision training

**Estimated Effort**: **1-2 days**

---

### 🟢 LOW - Nice to Have

#### 7. Clean Up Unused Code
**Priority**: **LOW** - Don't do until pipeline is complete

**Tasks**:
- Delete `main.py` (or repurpose)
- Move `devel.py`, `explore.py` to `src/dev/` or `tools/`
- Consider consolidating spectrogram methods

**Estimated Effort**: **4 hours**

---

#### 8. Documentation
**Priority**: **LOW**

**Tasks**:
- Update README.md with architecture overview
- Add docstrings to remaining functions
- Create usage examples in docs/
- Consider Sphinx for API docs

**Estimated Effort**: **2 days**

---

## 🎯 Recommended Implementation Order

### Phase 1: Core ML Pipeline (Week 1)
1. ✅ **Day 1-2**: Implement `models.py`, `losses.py`, `metrics.py`
2. ✅ **Day 3-4**: Implement `train.py` with training loop
3. ✅ **Day 5**: Implement `evaluate.py`
4. ✅ **Day 5**: Fix config inconsistencies
5. ✅ **Test**: Run first training experiment on small subset

### Phase 2: Enhancements (Week 2)
6. **Day 6**: Add data augmentation
7. **Day 7-8**: Full training run + hyperparameter tuning
8. **Day 9-10**: Testing infrastructure

### Phase 3: Polish (Week 3)
9. **Day 11-12**: Code organization refactor
10. **Day 13-14**: Documentation
11. **Day 15**: Final cleanup

---

## ✅ Quick Wins (Do Immediately)

These can be done in parallel with Phase 1:

1. **Fix Makefile**: Standardize to `uv run python` (10 min)
2. **Delete cnn.yaml**: It's empty and unused (1 min)
3. **Remove features section**: From baseline.yaml (2 min)
4. **Create placeholder files**: Touch `src/train.py`, `src/evaluate.py` to prevent import errors (1 min)

---

## ⚠️ Critical Design Decisions Needed

Before implementing Phase 1, decide:

1. **Model Architecture**:
   - Standard CNN (VGG-style)?
   - ResNet-style with skip connections?
   - Vision Transformer?
   - Pre-trained backbone (e.g., PANNs)?

2. **Feature Representation**:
   - Stick with SuperGaussian spectrogram?
   - Add mel spectrogram baseline for comparison?
   - Both (abstraction layer)?

3. **Training Strategy**:
   - Train on curated only?
   - Pre-train on noisy, fine-tune on curated?
   - Joint training with noise-robust loss?
   - Self-supervised pre-training (auto-encoding)?

4. **Evaluation Protocol**:
   - Use official Kaggle split?
   - Create own validation split from train_curated?
   - K-fold cross-validation?

---

## 📊 Summary

**Current State**: **60% Complete**
- ✅ Data pipeline: 100%
- ✅ Feature extraction: 100%
- ✅ Exploration tools: 100%
- ❌ Model architecture: 0%
- ❌ Training pipeline: 0%
- ❌ Evaluation: 0%
- ❌ Testing: 0%

**Code Quality**: **Good**
- Type hints: ✅
- Docstrings: ✅ (mostly)
- Linting: ✅ (ruff configured)
- Modularity: ✅
- Performance: ✅ (optimized)

**Blockers**: **3 Critical**
1. No model implementation
2. No training loop
3. No evaluation metrics

**Estimated Time to MVP**: **1-2 weeks** (assuming full-time work)

---

**Next Steps**: Read dataset documentation and analyze state-of-the-art approaches before implementing ML pipeline. Consider Vision Transformer with auto-encoding for noisy label handling.