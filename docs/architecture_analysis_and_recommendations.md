# Architecture Analysis and Recommendations
**FSDKaggle2019 Audio Tagging with Noisy Labels**

**Date**: 2026-03-29
**Author**: Analysis based on dataset documentation and state-of-the-art research (2021-2025)

---

## Table of Contents
1. [Dataset Analysis](#1-dataset-analysis)
2. [Challenge Overview](#2-challenge-overview)
3. [State-of-the-Art Review](#3-state-of-the-art-review)
4. [Architecture Options](#4-architecture-options)
5. [Your Proposal Analysis](#5-your-proposal-analysis-vision-transformer-with-auto-encoding)
6. [Leveraging Signal Processing Expertise](#6-leveraging-signal-processing-expertise)
7. [Final Recommendations](#7-final-recommendations)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Dataset Analysis

### 1.1 FSDKaggle2019 Overview

**Source**: DCASE 2019 Challenge Task 2 - "Audio tagging with noisy labels and minimal supervision"

**Dataset Structure:**
```
Curated Train:  4,970 clips  (~10.5 hours)   - Clean labels (correct but incomplete)
Noisy Train:    19,815 clips (~80 hours)     - Noisy labels (automatic annotation)
Test:           4,481 clips  (~12.9 hours)   - Clean labels (correct and complete)
─────────────────────────────────────────────────────────────────────────────
Total:          29,266 clips (~103.4 hours)  - 80 classes (AudioSet Ontology)
```

**Key Characteristics:**
- **Multi-label classification**: 80 classes, average 1.2 labels/clip
- **Label quality distribution**: 80% noisy / 20% curated (at clip level)
- **Acoustic mismatch**:
  - Curated + Test: Freesound (high quality, purpose-recorded)
  - Noisy: Flickr video soundtracks (lower quality, incidental audio)
- **Variable duration**: 0.3-30s (curated), 1-15s (noisy, mostly 15s)
- **Audio format**: PCM 16-bit, 44.1 kHz, mono

**Known Issues:**
- 5 files with wrong labels (curated set): `f76181c4.wav`, `77b925c2.wav`, `6a1f682a.wav`, `c7db12aa.wav`, `7752cc8a.wav`
- 1 corrupted file (curated set): `1d44b0bd.wav`

### 1.2 Challenge Characteristics

This dataset presents **three distinct challenges** that must be addressed:

#### A. Label Noise (Primary Challenge)
- **Noisy train set**: Automatically generated labels with substantial errors
  - In-vocabulary noise: Wrong class from the 80 classes
  - Out-of-vocabulary noise: Sounds not in the 80-class vocabulary
  - Missing labels: Incomplete annotations
- **Noise varies by class**: Some categories have much noisier labels than others

#### B. Limited Supervision
- Only **4,970 clean samples** for supervised learning
- Must leverage **19,815 noisy samples** effectively
- ~62 clean samples per class on average (some classes have fewer)

#### C. Domain Shift
- Training (noisy): Flickr video soundtracks
- Training (curated) + Test: Freesound recordings
- **Acoustic mismatch** between sources affects transfer

---

## 2. Challenge Overview

### 2.1 Official Metric

**Primary Metric**: **lwlrap** (label-weighted label-ranking average precision)
- Variant of mAP adapted for multi-label classification
- Accounts for label imbalance
- Values range from 0 to 1 (higher is better)

### 2.2 DCASE 2019 Results

**Top Kaggle Competition Results:**
- 1st place: 0.75980 lwlrap (private leaderboard)
- 2nd place: 0.75942 lwlrap
- 3rd place: 0.75787 lwlrap

**Note**: Competition ended in 2019, but dataset remains a relevant benchmark for noisy label learning.

---

## 3. State-of-the-Art Review

### 3.1 Vision Transformers for Audio (2021-2025)

#### Audio Spectrogram Transformer (AST) - 2021
**Paper**: "AST: Audio Spectrogram Transformer" (Gong et al., Interspeech 2021)

**Key Innovation**: First pure Vision Transformer for audio classification
- Treats spectrograms as images (patches of frequency × time)
- Pre-trained on ImageNet-21k (transfer learning from vision)
- Fine-tuned on audio datasets

**Results**:
- AudioSet: 0.485 mAP (state-of-the-art in 2021)
- ESC-50: 95.6% accuracy
- Speech Commands V2: 98.1% accuracy

**Architecture**:
- Input: 128 mel-bins × 1024 time frames (10s audio)
- Patch size: 16×16 → 12×64 = 768 patches
- 12-layer Transformer encoder (ViT-Base)
- Pre-training: ImageNet → AudioSet → target dataset

**Limitations**:
- Requires **large-scale pre-training** (ImageNet + AudioSet)
- Heavy computational cost (12-layer transformer)
- Pure transformer struggles with **limited labeled data**

---

#### Self-Supervised AST (SSAST) - 2022
**Paper**: "SSAST: Self-Supervised Audio Spectrogram Transformer" (Gong et al., AAAI 2022)

**Innovation**: Self-supervised pre-training for AST
- **Masked Autoencoding**: Mask 75% of spectrogram patches
- **Contrastive Learning**: Frame-level and clip-level objectives
- Pre-train on unlabeled audio, fine-tune on labeled

**Advantages**:
- No need for ImageNet pre-training
- Can leverage **unlabeled audio** (relevant for noisy dataset!)
- Better than supervised pre-training with limited labels

**Results**:
- AudioSet: 0.447 mAP (frozen encoder)
- Speech Commands: 97.4% accuracy

---

#### Recent Developments (2024-2025)

**EAT (Efficient Audio Transformer) - 2024** (IJCAI):
- Hybrid architecture: **Transformer encoder + lightweight CNN decoder**
- 80% masking ratio for self-supervised pre-training
- Combines **masked modeling** + **bootstrap learning**
- More efficient than pure AST

**FAST (Fast Audio Spectrogram Transformer) - 2025**:
- Optimizes AST for **speed and efficiency**
- CNN feature extraction + Transformer
- AudioSet: 0.448 mAP (slightly lower than AST but faster)

**Key Trend**: Hybrid CNN-Transformer architectures emerging as practical solution

---

### 3.2 DCASE 2019 Winning Techniques

#### 3rd/4th Place Team (Akiyama & Sato, 2019)
**Techniques**:
1. **Multitask Learning**:
   - Main task: Audio classification
   - Auxiliary task: Distinguish curated vs noisy samples
2. **Semi-supervised Learning**:
   - Use model predictions to relabel noisy samples
   - Iteratively refine labels
3. **Architecture**: VGGish (8 conv layers) + attention pooling
4. **Augmentation**: Mixup, SpecAugment
5. **Label Smoothing**: Combat overfitting to noisy labels

**Results**: 0.75787 lwlrap (3rd place)

---

#### Common DCASE 2019 Techniques

**Pre-processing**:
- Log-mel spectrograms (128 bins, 1-2s segments)
- Mixup augmentation (α=0.3-0.4): ~0.015-0.020 lwlrap improvement
- SpecAugment (time + frequency masking)

**Architectures**:
- CNN baselines: VGG, ResNet, DenseNet
- Attention mechanisms: Self-attention, global average pooling
- Ensemble methods: Multiple models + architectures

**Noisy Label Handling**:
1. **Label Smoothing**: Reduce confidence in noisy samples
2. **Sample Weighting**: Weight curated samples higher
3. **Co-teaching**: Two networks teach each other (filter noisy samples)
4. **Semi-supervised**: Use model predictions to relabel noisy data
5. **Curriculum Learning**: Start with curated, gradually add noisy

**Training Strategies**:
- Pre-train on noisy → fine-tune on curated
- Joint training with weighted loss
- Two-stage: Train on curated → adapt to noisy

---

### 3.3 CNN vs Transformer Comparison (2024 Research)

**Key Finding**: "Vision Transformers achieve 91% accuracy with spectrograms, while CNNs attain 95% accuracy" (2024 study)

**Implications**:
- **Transformers need more data**: AST requires ImageNet + AudioSet pre-training
- **CNNs more data-efficient**: Better with limited labeled data
- **Hybrid approaches winning**: CNN feature extraction + Transformer reasoning

**For FSDKaggle2019** (limited clean labels):
- Pure Transformer: Risky without massive pre-training
- CNN baseline: Safer, proven effective
- Hybrid CNN-Transformer: Best of both worlds

---

## 4. Architecture Options

### Option 1: CNN Baseline (Recommended Starting Point)

**Architecture**: ResNet-style or VGG-style CNN
```
Input: SuperGaussian Spectrogram (128 bands × T frames)
↓
Conv Block 1: [16 filters, 3×3, MaxPool 2×2]
↓
Conv Block 2: [32 filters, 3×3, MaxPool 2×2]
↓
Conv Block 3: [64 filters, 3×3, MaxPool 2×2]
↓
Global Average Pooling
↓
Dropout (0.4)
↓
Dense Layer: 80 classes, sigmoid
```

**Pros**:
- ✅ **Proven effective** on FSDKaggle2019 (DCASE 2019 winners used CNNs)
- ✅ **Data-efficient**: Works well with limited labels
- ✅ **Fast training**: Lower computational cost
- ✅ **Good inductive bias**: Translation invariance for audio
- ✅ **Easy to debug**: Well-understood architecture

**Cons**:
- ❌ Limited long-range dependencies
- ❌ No pre-training from large-scale data
- ❌ Less "impressive" for portfolio (standard approach)

**Use Case**: Establish strong baseline quickly

---

### Option 2: Audio Spectrogram Transformer (AST)

**Architecture**: Pure Vision Transformer on spectrograms
```
Input: SuperGaussian Spectrogram (128 bands × 1024 frames)
↓
Patch Embedding: 16×16 patches → 768 patches
↓
Positional Encoding
↓
12-layer Transformer Encoder (ViT-Base)
↓
Classification Head: 80 classes, sigmoid
```

**Pros**:
- ✅ **State-of-the-art potential**: AST achieved 0.485 mAP on AudioSet
- ✅ **Transfer learning**: Pre-trained weights available (Hugging Face)
- ✅ **Impressive for portfolio**: Modern architecture
- ✅ **Global receptive field**: Captures long-range dependencies

**Cons**:
- ❌ **Requires pre-training**: ImageNet or AudioSet (millions of samples)
- ❌ **Heavy computation**: 12-layer transformer is expensive
- ❌ **Data-hungry**: Poor performance with only 4,970 clean samples
- ❌ **Risk of failure**: Without proper pre-training, may underperform CNN

**Use Case**: If you can leverage pre-trained AST weights (Hugging Face)

---

### Option 3: Self-Supervised Pre-training + Fine-tuning

**Your Proposed Approach**: Auto-encoder pre-training on noisy data

**Two-Stage Training**:

**Stage 1: Self-supervised pre-training (on all data)**
```
Encoder: CNN or Transformer
↓
Masked Spectrogram Autoencoding
↓
Reconstruct masked patches
↓
Learn representations without labels
```

**Stage 2: Supervised fine-tuning (on curated data)**
```
Freeze/fine-tune encoder
↓
Add classification head
↓
Train on clean labels
```

**Pros**:
- ✅ **Leverage noisy data**: Use all 19,815 noisy samples for representation learning
- ✅ **No external pre-training**: Learn from your own data
- ✅ **Noise-robust**: Auto-encoding doesn't rely on noisy labels
- ✅ **Novel approach**: Good for portfolio (shows creativity)
- ✅ **Aligns with research**: SSAST, EAT use similar ideas (2022-2024)

**Cons**:
- ❌ **Complex implementation**: Two-stage training pipeline
- ❌ **Hyperparameter tuning**: Masking ratio, learning rates, etc.
- ❌ **No guarantees**: May not outperform supervised baseline
- ❌ **Time-consuming**: More experiments needed

**Use Case**: After CNN baseline, explore self-supervised approach

---

### Option 4: Hybrid CNN-Transformer (Emerging Best Practice)

**Architecture**: Efficient feature extraction + global reasoning
```
Input: SuperGaussian Spectrogram
↓
CNN Feature Extractor (lightweight)
  - 3 conv blocks → 64 feature maps
↓
Transformer Encoder (4-6 layers, lightweight)
  - Patch features → global reasoning
↓
Classification Head: 80 classes, sigmoid
```

**Pros**:
- ✅ **Best of both worlds**: CNN efficiency + Transformer capacity
- ✅ **Data-efficient**: CNN inductive bias helps
- ✅ **Modern architecture**: EAT, FAST use this approach (2024-2025)
- ✅ **Flexible**: Can adjust CNN/Transformer ratio

**Cons**:
- ❌ More complex than pure CNN
- ❌ More hyperparameters to tune

**Use Case**: After CNN baseline, upgrade to hybrid

---

## 5. Your Proposal Analysis: Vision Transformer with Auto-encoding

### 5.1 Proposal Interpretation

Based on your statement:
> "I was thinking about implementing a visual transformer, with possibly an auto-encoding part as the major part of the dataset has wrong annotations."

**Interpretation**:
- **Stage 1**: Self-supervised pre-training (auto-encoder) on **noisy + curated** data
- **Stage 2**: Supervised fine-tuning on **curated** data only

This is a **self-supervised learning approach** leveraging the noisy data for representation learning without trusting its labels.

---

### 5.2 Feasibility Analysis

#### ✅ Strengths of Your Proposal

**1. Addresses Label Noise Problem**
- Auto-encoding doesn't use labels → immune to label noise
- Can leverage all 24,785 samples (noisy + curated) for pre-training
- Aligns with **SSAST** (2022) and **EAT** (2024) approaches

**2. Leverages Unique Signal Processing**
- Your **SuperGaussian filter bank** is a differentiator
- Can use it as input to auto-encoder
- No one else has this representation for FSDKaggle2019

**3. Research-Backed**
- SSAST achieved strong results with masked autoencoding
- EAT (2024) combines auto-encoding + contrastive learning
- Self-supervised learning is a hot research topic

**4. Portfolio Value**
- Shows advanced understanding of ML (self-supervised learning)
- Demonstrates creativity (going beyond standard approaches)
- Combines signal processing + deep learning expertise

---

#### ⚠️ Challenges and Risks

**1. Implementation Complexity**
- Two-stage training pipeline (pre-train → fine-tune)
- More hyperparameters: masking ratio, reconstruction loss weight, etc.
- Longer training time (pre-train + fine-tune)

**2. No Guarantee of Success**
- Self-supervised methods require careful tuning
- May not outperform supervised CNN baseline
- Risk of wasted effort if approach doesn't work

**3. Transformer Data Requirements**
- Pure ViT still needs significant data even with pre-training
- 24,785 samples may be borderline for Transformer
- CNN encoder might be more reliable

**4. Evaluation Uncertainty**
- Unknown if self-supervised pre-training on this dataset is sufficient
- AudioSet (2M samples) vs FSDKaggle2019 (25k samples) is huge difference

---

### 5.3 Recommended Modifications

**Modified Proposal**: Hybrid Self-Supervised Approach

Instead of pure Vision Transformer, use:

**Stage 1: CNN-based Masked Autoencoding (on all data)**
```
Input: SuperGaussian Spectrogram (128 × T)
↓
CNN Encoder (ResNet-style)
  - Extract hierarchical features
↓
Mask random patches (50-75%)
↓
Lightweight CNN Decoder
  - Reconstruct masked patches
↓
Loss: MSE(reconstructed, original)
```

**Stage 2: Fine-tuning for Classification (on curated data)**
```
Frozen/Fine-tune CNN Encoder
↓
Add Global Average Pooling
↓
Classification Head: Dense(80, sigmoid)
↓
Loss: BCEWithLogitsLoss(predictions, labels)
```

**Why CNN instead of pure Transformer?**
- ✅ More data-efficient (important with only 25k samples)
- ✅ Faster training
- ✅ Better inductive bias for spectrograms
- ✅ Lower risk of failure

**Why not pure Transformer?**
- ❌ 25k samples likely insufficient for pure ViT
- ❌ AST required ImageNet (14M images) + AudioSet (2M samples)
- ❌ Risk of underperforming CNN baseline

---

## 6. Leveraging Signal Processing Expertise

### 6.1 Your Unique Advantage: SuperGaussian Filter Bank

**What You Have**: A custom, analytically designed filter bank with:
- Order-4 Gaussian (f⁴) frequency selectivity
- Dual-range geometric spacing (f_min, f_mid, f_max)
- Multi-resolution processing (adaptive downsampling)
- Pre-computed kernels for efficiency

**Why This Matters**:
1. **Novel representation**: No one else has used SuperGaussian for FSDKaggle2019
2. **Steeper roll-off**: Better frequency localization than standard mel
3. **Tunable**: Can optimize f_mid for this dataset
4. **Efficient**: Pre-computed, optimized for M4 GPU

**Comparison to Standard Approaches**:
- DCASE 2019 winners: Standard mel spectrograms (128 bins)
- AST papers: Standard mel spectrograms
- **You**: SuperGaussian with dual-range spacing → potential advantage

---

### 6.2 Feature Engineering Opportunities

#### A. Exploit Dual-Range Design

**Current**: f_mid = 1000 Hz (arbitrary choice)

**Optimization Opportunity**:
- Experiment with different f_mid values (500 Hz, 1500 Hz, 2000 Hz)
- Analyze per-class frequency distributions (you have spectrograms!)
- Choose f_mid to maximize discrimination for 80 classes

**Hypothesis**: Optimal f_mid may differ from 1000 Hz for this dataset

---

#### B. Multi-Scale Features

**Idea**: Leverage multi-resolution processing in your filter bank
- **Low resolution** (downsampled): Capture broad frequency trends
- **High resolution** (full rate): Capture fine frequency details

**Implementation**:
```python
# Extract features at multiple resolutions
features_level_0 = filter_bank.compute_at_level(waveform, level=0)  # Full res
features_level_2 = filter_bank.compute_at_level(waveform, level=2)  # 4× downsampled
features_level_4 = filter_bank.compute_at_level(waveform, level=4)  # 16× downsampled

# Concatenate or fuse
multi_scale_features = torch.cat([features_level_0, features_level_2, features_level_4], dim=1)
```

**Benefit**: Captures both local and global frequency patterns

---

#### C. Compare Against Mel Baseline

**Portfolio Value**: Show that SuperGaussian outperforms mel

**Experiment**:
1. Train CNN with **mel spectrograms** (baseline)
2. Train CNN with **SuperGaussian spectrograms** (your method)
3. Compare lwlrap scores

**Expected Outcome**:
- If SuperGaussian wins → strong portfolio piece ("I designed a better representation")
- If mel wins → learn why, iterate on filter bank

---

### 6.3 Acoustic Engineering Insights

**Your Background**: Acoustic engineer + signal processing expert

**Leverage This**:
1. **Frequency Analysis**:
   - Inspect spectrograms per class
   - Identify discriminative frequency ranges
   - Design class-specific features if needed

2. **Noise Robustness**:
   - Flickr soundtracks have background noise
   - Can you design noise-robust features?
   - Pre-processing: spectral subtraction, Wiener filtering?

3. **Temporal Dynamics**:
   - Audio events have characteristic temporal patterns
   - Can you design features that capture attack, sustain, decay?

---

## 7. Final Recommendations

### 7.1 Recommended Approach: Incremental Development

**Phase 1: Establish CNN Baseline (Week 1)**
```
Goal: Get a working system quickly
Architecture: Standard CNN (ResNet-style)
Input: SuperGaussian spectrograms (your filter bank)
Training: Supervised on curated data only
Expected: 0.65-0.70 lwlrap (rough estimate)
```

**Why Start Here**:
- ✅ Low risk, proven approach
- ✅ Fast to implement (follows cleanup plan Phase 1)
- ✅ Provides baseline for comparison
- ✅ Validates data pipeline and evaluation

**Deliverable**: Working training pipeline + evaluation metrics

---

**Phase 2: Noise-Robust Training (Week 2)**
```
Goal: Leverage noisy data effectively
Techniques:
  1. Sample weighting (curated > noisy)
  2. Label smoothing for noisy samples
  3. Mixup augmentation
  4. Two-stage training (noisy → curated)
Expected: +0.02-0.04 lwlrap improvement
```

**Why This Next**:
- ✅ Addresses core challenge (noisy labels)
- ✅ Proven techniques from DCASE 2019
- ✅ Low implementation complexity
- ✅ Leverages all data

**Deliverable**: Improved baseline with noise-robust training

---

**Phase 3: Self-Supervised Pre-training (Week 3-4) [OPTIONAL]**
```
Goal: Explore your proposal (auto-encoding)
Architecture: CNN encoder-decoder
Pre-training: Masked reconstruction on all 25k samples
Fine-tuning: Classification on curated data
Expected: Uncertain (research question)
```

**Why This is Optional**:
- ⚠️ Higher risk, uncertain payoff
- ⚠️ More complex implementation
- ✅ Good portfolio piece if it works
- ✅ Research contribution

**Decision Point**: Only proceed if Phase 2 results are promising

---

**Phase 4: Hybrid CNN-Transformer (Week 5) [ADVANCED]**
```
Goal: State-of-the-art architecture
Architecture: CNN feature extractor + lightweight Transformer
Training: Can use self-supervised pre-training from Phase 3
Expected: Potential SOTA if done well
```

**Why This is Advanced**:
- Requires Phase 1-2 working well
- More hyperparameters to tune
- Aligns with 2024-2025 research trends

---

### 7.2 Architecture Recommendation Matrix

| Architecture | Data Efficiency | Training Time | SOTA Potential | Risk | Portfolio Value |
|--------------|-----------------|---------------|----------------|------|-----------------|
| **CNN Baseline** | ✅✅✅ High | ✅✅✅ Fast | ⚠️ Medium | ✅ Low | ⚠️ Standard |
| **CNN + Noise-Robust** | ✅✅✅ High | ✅✅ Medium | ✅✅ Good | ✅ Low | ✅ Good |
| **Self-Supervised CNN** | ✅✅ Good | ⚠️ Slow | ✅✅ Good | ⚠️ Medium | ✅✅✅ Excellent |
| **Pure Transformer (AST)** | ❌ Poor | ❌ Very Slow | ✅✅✅ Excellent* | ❌ High | ✅✅ Good |
| **Hybrid CNN-Transformer** | ✅✅ Good | ⚠️ Slow | ✅✅✅ Excellent | ⚠️ Medium | ✅✅✅ Excellent |

*Requires pre-trained weights

---

### 7.3 Final Recommendation

**For Your Project** (Portfolio + Learning + Applied AI):

**Recommended Path**:
1. ✅ **Start with CNN Baseline** (Week 1)
   - Use SuperGaussian features
   - Supervised training on curated data
   - Establish solid baseline

2. ✅ **Add Noise-Robust Training** (Week 2)
   - Leverage noisy data
   - Implement DCASE 2019 techniques
   - Improve baseline

3. ⚠️ **Self-Supervised Pre-training** (Week 3-4) - Your Proposal
   - CNN-based masked autoencoding
   - Pre-train on all data
   - Fine-tune on curated data
   - **Decision point**: Continue if promising

4. 🎯 **Final Architecture**: Hybrid CNN-Transformer (Week 5)
   - If self-supervised works: Use pre-trained CNN encoder
   - Add lightweight Transformer on top
   - State-of-the-art attempt

**Alternative Path** (Lower Risk):
1. ✅ CNN Baseline → CNN + Noise-Robust → Hybrid CNN-Transformer
2. ❌ Skip self-supervised pre-training
3. ✅ Use transfer learning from pre-trained AST if needed

---

## 8. Implementation Roadmap

### 8.1 Week-by-Week Plan

#### Week 1: CNN Baseline
- [ ] Implement CNN architecture (`src/models.py`)
- [ ] Implement loss functions (`src/losses.py`)
- [ ] Implement metrics (`src/metrics.py`)
- [ ] Implement training loop (`src/train.py`)
- [ ] Implement evaluation (`src/evaluate.py`)
- [ ] Train on curated data only
- [ ] Evaluate on test set
- [ ] **Milestone**: Baseline lwlrap score

**Expected Result**: 0.65-0.70 lwlrap (rough estimate)

---

#### Week 2: Noise-Robust Training
- [ ] Implement sample weighting (curated > noisy)
- [ ] Add label smoothing for noisy samples
- [ ] Implement mixup augmentation
- [ ] Implement SpecAugment (time/frequency masking)
- [ ] Two-stage training: noisy → curated
- [ ] Hyperparameter tuning
- [ ] **Milestone**: Improved lwlrap

**Expected Result**: +0.02-0.04 lwlrap improvement

---

#### Week 3-4: Self-Supervised Pre-training (Optional)
- [ ] Design masked reconstruction objective
- [ ] Implement encoder-decoder architecture
- [ ] Pre-train on all 25k samples
- [ ] Evaluate learned representations
- [ ] Fine-tune on curated data
- [ ] Compare to supervised baseline
- [ ] **Milestone**: Self-supervised model evaluation

**Decision Point**: Continue to Week 5 if results are promising

---

#### Week 5: Hybrid CNN-Transformer (Advanced)
- [ ] Design hybrid architecture
- [ ] Implement lightweight Transformer
- [ ] Integrate with CNN encoder (from Week 3 or baseline)
- [ ] Train and evaluate
- [ ] Hyperparameter optimization
- [ ] **Milestone**: Final model performance

**Target**: Approach or exceed DCASE 2019 winners (0.758 lwlrap)

---

### 8.2 Key Implementation Details

#### Feature Extraction
```python
# Use your existing MultiResolutionFilterBank
from src.spectrogram_optimized import MultiResolutionFilterBank

def extract_features(waveform, sample_rate, config):
    """Extract SuperGaussian spectrogram features."""
    filter_bank = MultiResolutionFilterBank(
        envelope_class=SuperGaussianEnvelope,
        f_min=config['f_min'],
        f_max=config['f_max'],
        f_mid=config['f_mid'],
        num_bands=config['n_bands'],
        sample_rate=sample_rate,
        signal_duration=config['signal_duration'],
    )

    spec, time_step, _ = filter_bank.compute_spectrogram(
        waveform,
        hop_length=config['hop_length']
    )

    return spec  # Shape: (n_bands, n_frames)
```

#### Data Loading
```python
# Modify AudioDataset to return spectrograms
class AudioSpectrogramDataset(Dataset):
    def __init__(self, config, dataset_type, use_noisy=False):
        self.config = config
        self.dataset_type = dataset_type
        self.use_noisy = use_noisy

        # Load metadata
        # Initialize filter bank

    def __getitem__(self, idx):
        # Load audio
        # Extract spectrogram
        # Apply augmentation
        # Return: {'spectrogram': spec, 'labels': labels, 'is_noisy': flag}
```

#### Noise-Robust Loss
```python
def noise_robust_loss(outputs, targets, is_noisy, alpha=0.1):
    """Weighted BCE loss with label smoothing for noisy samples."""
    # Label smoothing for noisy samples
    if is_noisy:
        targets = targets * (1 - alpha) + 0.5 * alpha

    # Sample weighting
    sample_weight = 1.0 if not is_noisy else 0.5

    loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    loss = (loss * sample_weight).mean()

    return loss
```

#### Masked Autoencoding (Self-Supervised)
```python
class MaskedAutoencoderCNN(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # Random masking
        mask = torch.rand(x.shape) < self.mask_ratio
        x_masked = x * ~mask

        # Encode
        features = self.encoder(x_masked)

        # Decode
        reconstructed = self.decoder(features)

        # Loss: only on masked patches
        loss = F.mse_loss(reconstructed[mask], x[mask])

        return loss
```

---

### 8.3 Evaluation Strategy

**Metrics to Track**:
1. **lwlrap** (primary metric)
2. **mAP** (mean average precision)
3. **Per-class F1 score** (identify weak classes)
4. **Precision/Recall** curves

**Evaluation Protocol**:
- Validate on curated train split (80/20)
- Final evaluation on full test set
- Report per-class performance
- Confusion matrix analysis

**Comparison Baselines**:
1. Supervised CNN on curated only (baseline)
2. Supervised CNN with mel features (standard approach)
3. Your CNN with SuperGaussian features
4. Self-supervised pre-trained CNN
5. Hybrid CNN-Transformer

---

### 8.4 Portfolio Presentation

**Key Points to Highlight**:
1. **Signal Processing Expertise**:
   - Custom SuperGaussian filter bank design
   - Dual-range geometric spacing optimization
   - Analytical approach to feature extraction

2. **ML Engineering**:
   - Handled noisy labels effectively
   - Implemented self-supervised learning
   - Production-quality code with tests

3. **Research Contribution**:
   - Novel feature representation (SuperGaussian)
   - Self-supervised pre-training on small dataset
   - Ablation studies showing what works

4. **Results**:
   - Comparison to DCASE 2019 winners
   - Ablation studies (mel vs SuperGaussian)
   - Per-class performance analysis

---

## 9. Conclusion

### 9.1 Summary of Recommendations

**Architecture**: Start with CNN, optionally upgrade to hybrid CNN-Transformer

**Your Proposal** (ViT + Auto-encoding):
- ✅ Good idea for self-supervised learning
- ⚠️ Use CNN encoder instead of pure ViT (more data-efficient)
- ✅ Proceed after establishing CNN baseline

**Key Advantages**:
- ✅ SuperGaussian filter bank is a unique differentiator
- ✅ Self-supervised pre-training can leverage noisy data
- ✅ Aligns with 2022-2024 research (SSAST, EAT)

**Risk Mitigation**:
- Start with simple CNN baseline
- Add complexity incrementally
- Have fallback options at each stage

---

### 9.2 Expected Outcomes

**Conservative Estimate**:
- CNN Baseline: 0.65-0.70 lwlrap
- + Noise-Robust: 0.67-0.72 lwlrap
- + Self-Supervised: 0.68-0.74 lwlrap
- + Hybrid Architecture: 0.70-0.76 lwlrap

**Comparison to DCASE 2019**:
- 3rd place: 0.758 lwlrap
- Your target: 0.70-0.76 lwlrap (competitive)

**Portfolio Value**: High
- Shows both signal processing and ML expertise
- Demonstrates ability to handle noisy data
- Novel approach (SuperGaussian features)
- Modern techniques (self-supervised learning)

---

### 9.3 Next Steps

1. ✅ **Review this analysis**
2. ✅ **Decide on approach** (incremental or direct to self-supervised)
3. ✅ **Start with Phase 1**: Implement CNN baseline
4. ✅ **Iterate based on results**

**Recommendation**: Follow the incremental path (Weeks 1-2-3-4) to minimize risk while keeping your self-supervised proposal as the main exploration direction.

---

## References

1. Fonseca et al., "Audio tagging with noisy labels and minimal supervision", DCASE 2019
2. Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021
3. Gong et al., "SSAST: Self-Supervised Audio Spectrogram Transformer", AAAI 2022
4. Chen et al., "EAT: Self-Supervised Pre-Training with Efficient Audio Transformer", IJCAI 2024
5. Naman, "FAST: Fast Audio Spectrogram Transformer", 2025
6. Akiyama & Sato, "DCASE 2019 Task 2: Multitask Learning, Semi-supervised Learning", DCASE 2019

---

**Document Status**: Complete
**Next Action**: Implement CNN baseline (Week 1)