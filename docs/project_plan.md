# Freesound Audio Tagging - Project Plan

**Status**: Phase 1 (Foundation) - In Progress
**Last Updated**: 2026-04-02

---

## Quick Links

- **Architecture & Tasks**: See `.blueprints/` directory
  - `.blueprints/OVERVIEW.md` - Project architecture summary
  - `.blueprints/TASK_QUEUE.md` - Current work queue
  - `.blueprints/tasks/` - Individual task specifications
  - `.blueprints/decisions/` - Architecture decision records
- **DIC CNN Analysis**: `docs/dic_cnn_architecture_analysis.md`
- **Implementation Summary**: `docs/implementation_summary.md`

---

## Project Goals

1. **Multi-label audio classification** with noisy labels (FSDKaggle2019 dataset)
2. **State-of-the-art techniques**: Efficient CNN inspired by DIC implementation
3. **Acoustic engineering expertise**: SuperGaussian filters + psychoacoustic normalization
4. **Portfolio piece**: Production-quality code demonstrating ML + signal processing skills

---

## Development Phases

### Phase 1: Foundation (Week 1) - IN PROGRESS ✅ 25%

**Goal**: Establish solid infrastructure

**Tasks**:
- [x] Project structure refactoring (feature-based organization)
- [ ] Configuration system (dataclasses replacing YAML dicts)
- [ ] Testing infrastructure (pytest, model/feature tests)

**Status**: Project structure complete, config system next

---

### Phase 2: Efficient CNN (Week 2) - NOT STARTED

**Goal**: Implement DIC-inspired efficient CNN (300-500k parameters)

**Key Components**:
- WeightNormalizedConv2d
- Squeeze-and-Excitation blocks
- DropConnect regularization
- Separable convolutions (depthwise + pointwise)
- Mish activation + GroupNorm

**Reference**: `.blueprints/tasks/005-009*.md`, `.blueprints/decisions/D001-D003.md`

---

### Phase 3: Data Augmentation (Week 3) - NOT STARTED

**Goal**: Comprehensive augmentation pipeline

**Augmentations**:
- **Spectrogram**: SpecAugment (time/freq masking), additive noise
- **Waveform**: Time stretch, pitch shift, noise, reverb, freq response perturbation
- **Mix**: Mixup (batch-level)

**Modes**: Light, Medium, Heavy (configurable via YAML)

**Reference**: `.blueprints/tasks/010-012*.md`

---

### Phase 4: Training & Evaluation (Week 4) - NOT STARTED

**Goal**: Train models, ablation studies, performance analysis

**Experiments**:
1. Baseline CNN vs Efficient CNN
2. No augmentation vs Light vs Medium vs Heavy
3. Ablations: Mish vs ReLU, SE blocks, weight normalization, SuperGaussian vs mel

**Metrics**: lwlrap (primary), mAP, F1, AUC-ROC

**Reference**: `.blueprints/tasks/013-014*.md`

---

### Phase 5: Advanced Features (Week 5+) - OPTIONAL

**Goal**: Noisy label handling, self-supervised learning, transformers

**Topics**:
- Sample weighting, label smoothing, co-teaching
- Masked autoencoding pre-training
- Hybrid CNN-Transformer

---

## Key Technical Decisions

See `.blueprints/decisions/` for detailed rationale:

| Decision | Rationale |
|----------|-----------|
| **D001: Separable convolutions** | 8.4× fewer parameters, SE blocks add <2% overhead |
| **D002: Psychoacoustic normalization** | 20 dB temporal masking + 60 dB global floor, leverages acoustic expertise |
| **D003: Mish activation** | +0.5-1.5% accuracy over ReLU, worth 15% compute cost |
| **D004: GroupNorm (LayerNorm)** | Batch-size independent, better for spectrograms |
| **D005: Configuration dataclasses** | Type safety, IDE autocomplete, validation |

---

## Current Architecture

### Input Pipeline
- Audio: 5-second clips @ 44.1 kHz
- Features: SuperGaussian filter bank (128 bands, 20-8000 Hz, log-spaced)
- Normalization: Two-stage (temporal masking + global floor)

### Models
- **Current**: Baseline CNN (~28k parameters)
- **Planned**: Efficient CNN (300-500k parameters, 10-20× more efficient than naive)

### Training
- Loss: BCE, Focal, Asymmetric, Weighted
- Metrics: lwlrap, mAP, F1, Precision, Recall, AUC-ROC
- Device: MPS (Apple Silicon M4)
- Optimizer: Adam with cosine annealing

---

## DIC CNN Architecture Reference

The efficient CNN is inspired by a successful DIC (Digital Image Correlation) CNN implementation that achieved excellent results with only 324k parameters. Key techniques:

### Core Components
1. **Weight Normalization**: Normalizes conv weights to unit L2 norm (improves stability)
2. **Separable Convolutions**: Expansion (4-6×) → Depthwise → SE → Pointwise
3. **Squeeze-and-Excitation**: Channel attention with reduction ratio 4
4. **Mish Activation**: Better gradient flow than ReLU
5. **Group Normalization**: LayerNorm (num_groups=1) for spatial data
6. **Drop Connect**: Regularization via channel dropping
7. **Skip Connections**: Concatenation (DenseNet-style)

### Parameter Efficiency Example
Standard Conv2d: 64 → 128, 3×3 kernel = 73,728 params
Separable: 64 × 9 + 64 × 128 = 8,768 params (**8.4× fewer**)

See `docs/dic_cnn_architecture_analysis.md` for complete details.

---

## Data Augmentation Strategy

### Augmentation Pipeline Order
1. Time stretch (±10%, waveform)
2. Pitch shift (±2 semitones, waveform)
3. Add noise (SNR 30-40 dB, waveform)
4. Reverberation (optional, expensive)
5. Frequency response perturbation (8 bands, ±3 dB)
6. → Extract spectrogram
7. SpecAugment (2 freq + 2 time masks, rectangular)
8. Mixup (batch-level, alpha=0.4)

### Modes
- **Light**: SpecAugment + Mixup + noise
- **Medium**: + time stretch + pitch shift
- **Heavy**: All augmentations

---

## Next Steps

**Immediate** (this week):
1. Implement configuration dataclasses (Task 002)
2. Set up pytest and write baseline tests (Tasks 003-004)
3. Review data augmentation parameters (see questions below)

**Upcoming** (next 2 weeks):
1. Implement efficient CNN components (Tasks 005-009)
2. Implement augmentation pipeline (Tasks 010-012)
3. Begin training experiments

---

## Open Questions

### Augmentation Parameters
1. **SpecAugment**: 2 freq + 2 time masks? Or 1+1?
2. **Time stretch**: ±10% (conservative) or ±20% (aggressive)?
3. **Pitch shift**: ±2 semitones (conservative) or ±5 semitones?
4. **Noise level**: SNR 40 dB (subtle) or SNR 30 dB (noticeable)?
5. **Freq perturbation**: 8 bands ±3 dB or 12 bands ±5 dB?
6. **Reverberation**: Implement but disable by default? Or skip entirely?
7. **Mixup domain**: Spectrograms (faster) or waveforms (before feature extraction)?

**Recommendation**: Start with conservative settings (marked above), tune based on results.

### Architecture Parameters
1. **Model size**: 300-500k target? Or smaller (~200k)? Or larger (~1M)?
2. **SE reduction**: 4 (standard)? Or 8 (lighter)? Or 2 (heavier)?
3. **Drop connect rate**: 0.2? Or 0.3? Or adaptive (increase with depth)?

---

## Progress Tracking

See `.blueprints/TASK_QUEUE.md` for detailed task status.

**Overall**: 7% (1/14 core tasks complete)

| Phase | Tasks Complete | Status |
|-------|---------------|--------|
| Phase 1 | 1/4 (25%) | 🔄 In Progress |
| Phase 2 | 0/5 (0%) | ⏳ Not Started |
| Phase 3 | 0/3 (0%) | ⏳ Not Started |
| Phase 4 | 0/2 (0%) | ⏳ Not Started |

---

## References

- **Dataset**: [FSDKaggle2019 (Zenodo)](https://zenodo.org/records/3612637)
- **DIC CNN**: `docs/dic_cnn_architecture_analysis.md`
- **Blueprints**: `.blueprints/` directory (tasks, decisions, overview)
- **Code**: See `CLAUDE.md` for coding standards

---

**Document Status**: ✅ Complete - Consolidated Summary
**Next Review**: After Phase 1 completion
