# Project Refactoring and Enhancement Plan

**Project**: Freesound Audio Tagging Benchmark
**Date Created**: 2026-04-01
**Status**: Planning Phase

---

## Table of Contents

---

1. [Implementation Roadmap](#implementation-roadmap)
2. [Progress Tracking](#progress-tracking)

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal**: Establish solid infrastructure

#### Tasks

1. **Project Structure Refactoring**
   - [x] Create new directory structure
   - [x] Move files to new locations
   - [x] Update all imports
   - [x] Test that existing scripts work

2. **Configuration System**
   - [ ] Implement `src/core/config.py` with dataclasses
   - [ ] Update `configs/baseline.yaml`
   - [ ] Replace all `dict` access with dot notation
   - [ ] Add configuration validation

3. **Testing Infrastructure**
   - [ ] Set up pytest
   - [ ] Create `tests/` structure
   - [ ] Write tests for existing models
   - [ ] Write tests for feature extraction
   - [ ] Set up CI/CD (optional)

**Deliverables**:
- ✅ Clean, modular project structure
- ✅ Type-safe configuration system
- ✅ Comprehensive test coverage (>80%)

**Priority**: **CRITICAL** - Must complete before other work

---

### Phase 2: Efficient CNN Architecture (Week 2)

**Goal**: Implement DIC-inspired efficient CNN

#### Tasks

1. **Model Components**
   - [ ] Implement `WeightNormalizedConv2d`
   - [ ] Implement `SqueezeAndExcitation`
   - [ ] Implement `DropConnect`
   - [ ] Implement `SepConvLayer`
   - [ ] Implement `SepConvBlock`
   - [ ] Write tests for all components

2. **Full Model**
   - [ ] Implement `EfficientAudioCNN` in `src/models/efficient_cnn.py`
   - [ ] Add weight normalization callback
   - [ ] Test forward pass, parameter count
   - [ ] Compare to baseline CNN

3. **Training Updates**
   - [ ] Add custom learning rate scheduler
   - [ ] Integrate weight normalization into training loop
   - [ ] Add mixed precision training (AMP)

**Deliverables**:
- ✅ Working `EfficientAudioCNN` with 300-500k parameters
- ✅ Tests passing for all components
- ✅ Training script updated

**Priority**: **HIGH** - Core improvement

---

### Phase 3: Data Augmentation (Week 3)

**Goal**: Implement comprehensive augmentation pipeline

#### Tasks

1. **Spectrogram Augmentations**
   - [ ] SpecAugment (time + frequency masking)
   - [ ] Additive noise (spectrogram domain)
   - [ ] Tests for spectrogram augmentations

2. **Waveform Augmentations**
   - [ ] Time stretching
   - [ ] Pitch shifting
   - [ ] Additive noise (waveform domain)
   - [ ] Frequency response perturbation
   - [ ] Tests for waveform augmentations

3. **Advanced Augmentations**
   - [ ] Reverberation (RIR generation + convolution)
   - [ ] Mixup (batch-level)
   - [ ] Tests for advanced augmentations

4. **Integration**
   - [ ] Create `AugmentationPipeline` class
   - [ ] Integrate into `Dataset` class
   - [ ] Add augmentation config to YAML
   - [ ] Test end-to-end with training

**Deliverables**:
- ✅ Complete augmentation library
- ✅ Configurable augmentation pipeline
- ✅ Tests for all augmentations

**Priority**: **HIGH** - Critical for performance

---

### Phase 4: Training and Evaluation (Week 4)

**Goal**: Train efficient model with augmentations, evaluate performance

#### Tasks

1. **Baseline Training**
   - [ ] Train baseline CNN (no augmentations)
   - [ ] Train efficient CNN (no augmentations)
   - [ ] Compare performance

2. **Augmented Training**
   - [ ] Train efficient CNN with light augmentations
   - [ ] Train efficient CNN with medium augmentations
   - [ ] Train efficient CNN with heavy augmentations

3. **Ablation Studies**
   - [ ] Mish vs ReLU
   - [ ] With/without SE blocks
   - [ ] With/without weight normalization
   - [ ] SuperGaussian vs mel spectrogram

4. **Evaluation and Analysis**
   - [ ] Compute metrics on test set
   - [ ] Per-class performance analysis
   - [ ] Generate visualizations
   - [ ] Update `implementation_summary.md`

**Deliverables**:
- ✅ Trained models with various configurations
- ✅ Performance comparison table
- ✅ Ablation study results
- ✅ Updated documentation

**Priority**: **MEDIUM** - Validation and optimization

---

### Phase 5: Advanced Features (Week 5+, Optional)

**Goal**: Explore advanced techniques

#### Tasks

1. **Noisy Label Handling**
   - [ ] Sample weighting
   - [ ] Label smoothing
   - [ ] Co-teaching
   - [ ] Use train_noisy data

2. **Self-Supervised Pre-training** (from architecture analysis)
   - [ ] Masked autoencoding
   - [ ] Pre-train on all data (curated + noisy)
   - [ ] Fine-tune on curated

3. **Hybrid CNN-Transformer** (if time permits)
   - [ ] Lightweight transformer encoder
   - [ ] CNN feature extractor + transformer
   - [ ] Compare to pure CNN

**Priority**: **LOW** - Advanced exploration

---

## Progress Tracking

### Legend

- ⏳ **Not Started**: Task not yet begun
- 🔄 **In Progress**: Currently working on task
- ✅ **Complete**: Task finished and tested
- ❌ **Blocked**: Task blocked by dependency or issue
- 🔍 **Review**: Task complete, needs review

---

### Current Status

**Overall Progress**: **3%** (Phase 1.1 complete - Project structure refactored)

---

### Phase 1: Foundation

| Task | Status | Assignee | Notes | Completed |
|------|--------|----------|-------|-----------|
| Create new directory structure | ✅ | Claude | Feature-based organization | 2026-04-02 |
| Move files to new locations | ✅ | Claude | All files migrated successfully | 2026-04-02 |
| Update imports | ✅ | Claude | All imports updated and tested | 2026-04-02 |
| Implement config dataclasses | ⏳ | - | High priority - Next task | - |
| Update YAML configs | ⏳ | - | Depends on config dataclasses | - |
| Replace dict access with dot notation | ⏳ | - | Depends on config dataclasses | - |
| Set up pytest | ⏳ | - | - | - |
| Write model tests | ⏳ | - | - | - |
| Write feature tests | ⏳ | - | - | - |

**Phase Progress**: 3/9 tasks complete (33%)

---

### Phase 2: Efficient CNN

| Task | Status | Assignee | Notes | Completed |
|------|--------|----------|-------|-----------|
| Implement WeightNormalizedConv2d | ⏳ | - | Copy from DIC | - |
| Implement SqueezeAndExcitation | ⏳ | - | Copy from DIC | - |
| Implement DropConnect | ⏳ | - | Copy from DIC | - |
| Implement SepConvLayer | ⏳ | - | Adapt from DIC | - |
| Implement SepConvBlock | ⏳ | - | Adapt from DIC | - |
| Write component tests | ⏳ | - | - | - |
| Implement EfficientAudioCNN | ⏳ | - | Main model | - |
| Test model forward pass | ⏳ | - | - | - |
| Add weight norm to training | ⏳ | - | - | - |
| Add custom LR scheduler | ⏳ | - | - | - |
| Add mixed precision (AMP) | ⏳ | - | For M4 GPU | - |

**Phase Progress**: 0/11 tasks complete (0%)

---

### Phase 3: Data Augmentation

| Task | Status | Assignee | Notes | Completed |
|------|--------|----------|-------|-----------|
| Implement SpecAugment | ⏳ | - | Time + freq masking | - |
| Implement spec noise | ⏳ | - | Gaussian noise | - |
| Implement time stretch | ⏳ | - | ±10% range | - |
| Implement pitch shift | ⏳ | - | ±2 semitones | - |
| Implement waveform noise | ⏳ | - | SNR 30-40 dB | - |
| Implement freq perturbation | ⏳ | - | 8 bands, ±3dB | - |
| Implement reverberation | ⏳ | - | RIR + fftconvolve | - |
| Implement Mixup | ⏳ | - | Batch-level | - |
| Create AugmentationPipeline | ⏳ | - | - | - |
| Integrate into Dataset | ⏳ | - | - | - |
| Write augmentation tests | ⏳ | - | - | - |

**Phase Progress**: 0/11 tasks complete (0%)

---

### Phase 4: Training and Evaluation

| Task | Status | Assignee | Notes | Completed |
|------|--------|----------|-------|-----------|
| Train baseline CNN | ⏳ | - | No augmentation | - |
| Train efficient CNN (baseline) | ⏳ | - | No augmentation | - |
| Train with light augmentation | ⏳ | - | SpecAug + Mixup | - |
| Train with medium augmentation | ⏳ | - | + time/pitch | - |
| Train with heavy augmentation | ⏳ | - | All augmentations | - |
| Ablation: Mish vs ReLU | ⏳ | - | - | - |
| Ablation: With/without SE | ⏳ | - | - | - |
| Ablation: Weight normalization | ⏳ | - | - | - |
| Ablation: SuperGaussian vs mel | ⏳ | - | - | - |
| Evaluate on test set | ⏳ | - | All models | - |
| Generate visualizations | ⏳ | - | - | - |
| Update documentation | ⏳ | - | - | - |

**Phase Progress**: 0/12 tasks complete (0%)

---

### Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-01 | Use dataclasses for configuration | Better IDE support, type safety |
| 2026-04-01 | Implement DIC-inspired efficient CNN | Excellent parameter efficiency |
| 2026-04-01 | Use Mish activation | Better than ReLU, worth compute cost |
| 2026-04-01 | Group normalization (num_groups=1) | Better for spectrograms than BatchNorm |
| 2026-04-01 | Reverberation optional (off by default) | Computationally expensive |
| 2026-04-01 | **Psychoacoustic temporal masking + global floor normalization** | Domain-informed feature engineering: 20 dB frame-wise masking removes imperceptible details (psychoacoustics), 60 dB global floor normalizes dynamic range (engineering). Complementary purposes, negligible compute cost (~0.05 ms), leverages acoustic engineering expertise |
| 2026-04-02 | **Feature-based project structure** | Reorganized from flat structure to feature-based organization (core/, data/, features/, models/, training/, visualization/, scripts/). Improves separation of concerns, testability, scalability. All imports updated and tested successfully. |

---

### Issues and Blockers

| Issue | Impact | Status | Resolution |
|-------|--------|--------|------------|
| - | - | - | - |

---

### Spectrogram Processing Notes

#### Overview: Combined Psychoacoustic Masking + Global Floor Normalization

This project uses a **two-stage normalization approach** that combines psychoacoustic temporal masking with global noise floor removal:

1. **Psychoacoustic temporal masking** (frame-wise, 20 dB): Removes imperceptible details based on human auditory masking
2. **Global noise floor** (60 dB): Establishes consistent dynamic range across all spectrograms
3. **Normalization**: Maps to [0, 1] range for neural network input

This approach leverages acoustic engineering expertise to create features that focus on perceptually relevant information.

---

#### Stage 1: Psychoacoustic Temporal Masking (Frame-wise)

**Scientific Basis**: Human auditory system cannot perceive sounds that are masked by louder sounds occurring simultaneously.

**Masking Threshold**: 20 dB below the loudest sound in each time frame
- **20 dB** = 100× intensity difference (10^(20/10) = 100)
- Humans cannot hear sounds this far below the masker
- Based on well-established psychoacoustic research (temporal masking)

**Purpose**:
- Remove imperceptible low-energy components within each frame
- Focus model on perceptually salient features
- Reduce effective noise, especially in dynamic signals (music, speech with pauses)

**When Most Effective**:
- Signals with high temporal dynamic range (loud/quiet passages)
- Non-uniform noise across time
- Multi-label classification requiring detection of multiple sounds at different levels

---

#### Stage 2: Global Noise Floor Removal

**Purpose**: Establish consistent dynamic range across all spectrograms

**Floor Reference Options**:
1. **Global max** (default): `-60 dB` from absolute maximum
2. **Percentile**: `-60 dB` from 95th percentile (more robust to outliers)
3. **RMS-based**: `-60 dB` from signal RMS level

**Dynamic Range**: 60 dB (6 Bell) total after floor removal
- Provides consistent input range for neural network
- Removes absolute noise floor
- Normalizes signals recorded at different levels

---

#### Stage 3: Normalization

**Linear Normalization** (recommended):
- Maps to [0, 1] range
- 0 dB (after floor removal) → 1.0
- floor_db (60 dB below max) → 0.0

**Standardization** (alternative):
- Zero mean, unit standard deviation
- May be preferred if model expects standardized inputs

---

#### Complete Implementation

```python
def normalize_spectrogram_db(
    spec_db: np.ndarray,  # Shape: (freq, time)
    config: SpectrogramNormalization
) -> np.ndarray:
    """
    Normalize spectrogram using psychoacoustic masking + noise floor removal.

    This combines two complementary approaches:
    1. Frame-wise temporal masking: Removes imperceptible details based on
       human auditory masking (sounds >20dB below loudest are inaudible)
    2. Global floor normalization: Removes absolute noise floor and
       normalizes to consistent dynamic range

    Args:
        spec_db: Spectrogram in dB (log magnitude), shape (freq, time)
        config: SpectrogramNormalization configuration

    Returns:
        Normalized spectrogram in [0, 1] range (or standardized)
    """
    spec_processed = spec_db.copy()

    # STAGE 1: Frame-wise temporal masking (psychoacoustic)
    if config.enable_temporal_masking:
        # Find reference level per time frame (95th percentile for robustness)
        if config.masking_reference == "percentile":
            ref_per_frame = np.percentile(
                spec_processed, config.masking_percentile, axis=0, keepdims=True
            )  # (1, time)
        else:  # "max"
            ref_per_frame = spec_processed.max(axis=0, keepdims=True)  # (1, time)

        # Set masking threshold (20 dB below frame reference)
        masking_threshold = ref_per_frame - config.masking_threshold_db

        # Set values below threshold to background level (NOT lift them up!)
        # This correctly implements psychoacoustic masking
        spec_processed = np.where(
            spec_processed < masking_threshold,
            config.background_level,  # Set to background level
            spec_processed  # Keep original value
        )

    # STAGE 2: Global noise floor removal
    if config.enable_global_floor:
        # Determine floor reference level
        if config.floor_reference == "global_max":
            floor_level = spec_processed.max() - config.floor_db
        elif config.floor_reference == "percentile":
            signal_level = np.percentile(spec_processed, config.percentile)
            floor_level = signal_level - config.floor_db
        elif config.floor_reference == "rms":
            # Use 95th percentile as robust signal level estimate
            signal_level = np.percentile(spec_processed, 95)
            floor_level = signal_level - config.floor_db
        else:
            raise ValueError(f"Unknown floor_reference: {config.floor_reference}")

        # Remove noise floor (set floor to 0)
        spec_processed = np.maximum(0, spec_processed - floor_level)

    # STAGE 3: Normalization
    if config.normalize_method == "linear":
        # Map to [0, 1]: 0 dB → 0, floor_db → 1
        if config.enable_global_floor:
            spec_norm = spec_processed / config.floor_db
        else:
            # Fallback if floor disabled
            max_val = spec_processed.max()
            spec_norm = spec_processed / (max_val + 1e-8)

    elif config.normalize_method == "standardize":
        # Z-score normalization (mean=0, std=1)
        mean = spec_processed.mean()
        std = spec_processed.std() + 1e-8
        spec_norm = (spec_processed - mean) / std

    elif config.normalize_method == "none":
        spec_norm = spec_processed
    else:
        raise ValueError(f"Unknown normalize_method: {config.normalize_method}")

    return spec_norm
```

---

#### Computational Cost Analysis

**Stage 1 (Temporal Masking)**:
- `max(axis=0)`: O(freq × time) = 128 × 500 = 64k ops ≈ 0.01 ms
- `Subtraction + maximum`: 64k ops ≈ 0.01 ms
- **Subtotal**: ~0.02 ms per spectrogram

**Stage 2 (Global Floor)**:
- `max()` or `percentile()`: O(N) ≈ 0.01 ms
- `Subtraction + maximum`: O(N) ≈ 0.01 ms
- **Subtotal**: ~0.02 ms per spectrogram

**Stage 3 (Normalization)**:
- `Division`: O(N) ≈ 0.01 ms
- **Subtotal**: ~0.01 ms per spectrogram

**Total Pipeline Cost**: ~0.05 ms per spectrogram

**Throughput**: ~20,000 spectrograms/second on single CPU core

**Verdict**: ✅ **Completely negligible** - suitable for real-time inference

---

#### Why Both Stages?

The two stages serve **complementary purposes** and work at different scales:

| Aspect | Temporal Masking (Frame-wise) | Global Floor |
|--------|-------------------------------|--------------|
| **Purpose** | Remove imperceptible details | Normalize dynamic range |
| **Basis** | Psychoacoustics (human perception) | Engineering (signal processing) |
| **Threshold** | 20 dB below frame max | 60 dB below global max |
| **Scale** | Per time frame (local) | Entire spectrogram (global) |
| **Adapts to** | Temporal dynamics | Recording level |

**Example**:
- **Temporal masking**: In a music piece with quiet verse + loud chorus, removes imperceptible details in *both* sections
- **Global floor**: Ensures the quiet verse and loud chorus are both represented in the same [0, 60] dB range

---

#### Recommended Configuration

**Default Settings** (SpectrogramNormalization):

```python
SpectrogramNormalization(
    # Temporal masking (psychoacoustic)
    enable_temporal_masking=True,
    masking_threshold_db=20.0,  # 2 Bell (20 dB)

    # Global floor
    enable_global_floor=True,
    floor_db=60.0,              # 6 Bell (60 dB total range)
    floor_reference="global_max",

    # Normalization
    normalize_method="linear"   # [0, 1] range
)
```

**Ablation Study Configurations**:

| Config | Temporal Masking | Global Floor | Purpose |
|--------|------------------|--------------|---------|
| Baseline | ❌ Off | ❌ Off | Raw spectrogram |
| Floor only | ❌ Off | ✅ On | Noise removal only |
| Masking only | ✅ On | ❌ Off | Psychoacoustic only |
| **Combined (recommended)** | ✅ On | ✅ On | Both approaches |

---

#### Implementation Notes

**Note 1**: "cB" vs "Bell"
- Graphs may show "cB" (centibel) - this is **incorrect**
- Should be "Bell": 1 Bell = 10 dB
- 2 Bell = 20 dB, 6 Bell = 60 dB

**Note 2**: Processing Order
1. Compute spectrogram (magnitude)
2. Convert to dB: `20 * log10(magnitude + epsilon)`
3. **Apply temporal masking** (Stage 1)
4. **Apply global floor** (Stage 2)
5. **Normalize** (Stage 3)
6. Pass to model

**Note 3**: Training vs Inference
- Both stages are deterministic (no randomness)
- Apply identically during training and inference
- No special handling needed for eval mode

---

#### Rationale Summary

**Why temporal masking?**
- ✅ Psychoacoustically grounded (based on human perception limits)
- ✅ Removes imperceptible noise, focuses on signal
- ✅ Domain expertise applied (acoustic engineering)
- ✅ Negligible computational cost (~0.02 ms)

**Why global floor?**
- ✅ Consistent dynamic range across recordings
- ✅ Normalizes to consistent input range for NN
- ✅ Removes absolute noise floor
- ✅ Standard signal processing practice

**Why both?**
- ✅ Complementary purposes (perceptual + engineering)
- ✅ Work at different scales (local + global)
- ✅ Total cost still negligible (~0.05 ms)
- ✅ Expected to improve both training and results

---

### Questions for Discussion

#### Spectrogram Processing

1. **Dynamic range cropping**:
   - Enable by default? ✅ **Recommended** (2 Bell frame-wise)
   - Use sliding window variant? **To explore** (may help for variable-loudness recordings)
   - Optimal crop levels?
     - 2 Bell (20 dB) frame-wise? ✅ **Recommended**
     - 6 Bell (60 dB) sliding window? **Optional**
   - Window size for sliding variant? **50 frames (~0.5s)?**

2. **Interaction with normalization**:
   - Apply cropping before or after normalization?
   - **Recommendation**: After converting to dB, before normalization

#### Data Augmentation

1. **SpecAugment masking pattern**:
   - Random rectangular patches? ✅ **Recommended**
   - Random points?
   - Random small/medium patches?

2. **Spectrogram noise masking value**:
   - Zero (silence)? ✅ **Recommended**
   - Mean value?
   - Random value?

3. **Time stretch range**:
   - Conservative (±10%)? ✅ **Recommended**
   - Aggressive (±20%)?

4. **Pitch shift range**:
   - Conservative (±2 semitones)? ✅ **Recommended**
   - Aggressive (±5 semitones)?

5. **Waveform noise level**:
   - SNR 40 dB (barely noticeable)?
   - SNR 30-40 dB? ✅ **Recommended**
   - SNR 20 dB (clearly audible)?

6. **Frequency response perturbation**:
   - 4-8 bands, ±2 dB?
   - 8 bands, ±3 dB? ✅ **Recommended**
   - 12-16 bands, ±5 dB?

7. **Reverberation**:
   - Implement but disable by default? ✅ **Recommended**
   - Skip entirely (too expensive)?

8. **Mixup domain**:
   - Spectrograms (faster)? ✅ **Recommended**
   - Waveforms (before feature extraction)?

9. **Augmentation for masked autoencoding** (future):
   - Use augmentation during pre-training?
   - Different augmentations for pre-training vs fine-tuning?

#### Architecture

10. **Model size target**:
    - 300-500k parameters? ✅ **Recommended**
    - Smaller (~200k)?
    - Larger (~1M)?

11. **SE reduction ratio**:
    - 4 (standard)? ✅ **Recommended**
    - 8 (lighter)?
    - 2 (heavier)?

12. **Drop connect rate**:
    - 0.2? ✅ **Recommended**
    - 0.3?
    - Adaptive (increase with depth)?

---

### Next Steps

**Immediate Actions** (this week):

1. **Review this plan** with user
2. **Discuss data augmentation questions** (above)
3. **Begin Phase 1**: Project structure refactoring
   - Create new directory structure
   - Implement configuration dataclasses
   - Set up testing infrastructure

**Upcoming** (next 1-2 weeks):

1. **Complete Phase 1** (foundation)
2. **Begin Phase 2** (efficient CNN)
3. **Start Phase 3** (augmentation)

---

## Relationship to `implementation_summary.md`

### Redundancy Analysis

**`implementation_summary.md`** serves as:
- Historical record of what was built (Phase 1)
- Snapshot of project state at a point in time
- Reference for completed work

**This document** (`project_plan_and_tracking.md`) serves as:
- Living plan for future work
- Progress tracker with status updates
- Decision log and discussion forum

### Recommendation

**Keep both files with distinct purposes**:

1. **`implementation_summary.md`**:
   - Rename to `phase1_implementation_summary.md`
   - Keep as historical record
   - Don't update further (frozen)

2. **`project_plan_and_tracking.md`** (this file):
   - Use for planning Phases 2-5
   - Update progress regularly
   - Log decisions and discussions

3. **New file: `project_status.md`** (optional):
   - High-level status summary
   - Links to detailed docs
   - Updated weekly

---

## Summary

This document provides a comprehensive plan for refactoring and enhancing the Freesound Audio Tagging project. The plan is organized into 5 phases:

1. **Foundation** (Week 1): Project structure, configuration, testing
2. **Efficient CNN** (Week 2): DIC-inspired architecture
3. **Data Augmentation** (Week 3): Comprehensive augmentation pipeline
4. **Training & Evaluation** (Week 4): Train models, evaluate, ablation studies
5. **Advanced Features** (Week 5+): Noisy labels, self-supervised learning, transformers

**Key innovations**:
- **DIC-inspired CNN**: 300-500k parameters with SOTA techniques
- **Dataclass configuration**: Type-safe, IDE-friendly
- **Comprehensive augmentation**: 8+ augmentation strategies
- **Modular architecture**: Feature-based organization

**Next action**: Implement configuration dataclasses (Phase 1.2)

---

**Document Status**: 🔄 In Progress - Phase 1.1 Complete
**Last Updated**: 2026-04-02
**Next Review**: After Phase 1 completion