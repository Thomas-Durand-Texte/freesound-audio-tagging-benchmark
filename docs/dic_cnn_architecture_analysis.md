# DIC CNN Architecture Analysis

**Source**: `/tmp/CNN_DIC/Networks.py` and `/tmp/training-4.ipynb`
**Date**: 2026-04-01
**Purpose**: Extract efficient CNN design patterns for audio tagging

---

## Overview

The Digital Image Correlation (DIC) CNN achieves **exceptional parameter efficiency** with only **~324k parameters** while maintaining high precision. This analysis extracts key architectural insights applicable to audio tagging.

---

## Key Architectural Innovations

### 1. Weight Normalization

**Class**: `WeightNormalizedConv2d`

**Mechanism**: Normalizes filter weights to unit L2 norm during training

```python
def weight_scaling(self):
    with torch.no_grad():
        self.weight *= 1. / torch.sqrt(
            (self.weight**2).sum(dim=(1, 2, 3), keepdim=True) + 1e-5
        )
```

**Benefits**:
- Stabilizes training (reduces internal covariate shift)
- Improves convergence speed
- Acts as implicit regularization
- Can be applied periodically (e.g., every N optimizer steps)

**Usage**: Call `model.apply_weight_scaling()` after optimizer step

---

### 2. Group Normalization (Layer Norm)

**Configuration**: `{'which': 'group', 'num_groups': 1}`

**Rationale**:
- `num_groups=1` ⟹ Full layer normalization
- Batch-independent (works with batch_size=1)
- Better for spectrograms (preserves spatial structure)
- More stable than BatchNorm for variable-length sequences

**Comparison**:
| Normalization | Scope | Batch-dependent | Best for |
|--------------|-------|-----------------|----------|
| BatchNorm | Across batch | ✅ Yes | Large batches, fixed shapes |
| GroupNorm (G=1) | Across all channels | ❌ No | Small batches, variable shapes |
| InstanceNorm | Per channel | ❌ No | Style transfer |

---

### 3. Mish Activation Function

**Formula**: `Mish(x) = x × tanh(softplus(x))`

**Properties**:
- Smooth, non-monotonic
- Self-regularizing
- Better gradient flow (no dead neurons)
- Empirically outperforms ReLU/Swish

**Performance**:
- ~10-15% slower than ReLU
- **Significant accuracy improvement** (worth the cost)

**Recommendation**: Use `nn.Mish(inplace=True)` for all activations

---

### 4. Separable Convolutions (MobileNet-style)

**Architecture**: `SepConvLayer`

**Pipeline**:
```
Input (C_in channels)
    ↓
[1] Expansion: 1×1 conv → C_in × expansion channels
    ↓
[2] Depthwise: 3×3 conv, groups=C_in × expansion
    ↓
[3] Squeeze-Excitation: Channel attention
    ↓
[4] Pointwise: 1×1 conv → C_out channels
    ↓
Output (C_out channels)
```

**Parameters**:
- `expansion`: Typically 4-6 (increases capacity without many parameters)
- `nFilterPerMap`: Usually 1 (number of filters per depthwise channel)
- `se`: Squeeze-and-Excitation enabled (True recommended)

**Parameter Efficiency**:

Standard conv: `C_in × C_out × 3 × 3 = 9 × C_in × C_out`

Separable conv (expansion=4):
- Expansion: `C_in × (4 × C_in) × 1 × 1 = 4 × C_in²`
- Depthwise: `(4 × C_in) × 1 × 3 × 3 = 36 × C_in`
- Pointwise: `(4 × C_in) × C_out × 1 × 1 = 4 × C_in × C_out`
- **Total**: `4 × C_in² + 36 × C_in + 4 × C_in × C_out`

**Savings**: For C_in ≈ C_out = 64:
- Standard: 36,864 params
- Separable: ~17,000 params
- **~50% reduction**

---

### 5. Squeeze-and-Excitation (SE) Blocks

**Mechanism**: Channel-wise attention

```python
def forward(self, x):  # x: (B, C, H, W)
    # 1. Squeeze: Global average pooling
    z = x.mean((-2, -1), keepdim=True)  # (B, C, 1, 1)

    # 2. Excitation: FC → ReLU → FC → Sigmoid
    s = FC_reduce(z)   # (B, C, 1, 1) → (B, C/r, 1, 1)
    s = ReLU(s)
    s = FC_expand(s)   # (B, C/r, 1, 1) → (B, C, 1, 1)
    s = Sigmoid(s)     # Values in [0, 1]

    # 3. Scale: Element-wise multiplication
    return x * s
```

**Parameters**:
- Reduction ratio `r = 4` (standard)
- Adds: `2 × C²/r = C²/2` parameters
- **Overhead**: ~1-2% parameters
- **Benefit**: +0.5-1.0% accuracy

**Key insight**: Learns to emphasize important frequency bands (for audio)

---

### 6. Drop Connect (Stochastic Depth)

**Class**: `DropConnect`

**Mechanism**: Randomly drop entire channels during training

```python
def forward(self, x):
    if self.training:
        keep_prob = 1 - drop_rate
        random_tensor = keep_prob + torch.rand((x.shape[0], x.shape[1], 1, 1))
        binary_mask = torch.floor(random_tensor)
        return x * (binary_mask / keep_prob)
    return x
```

**Benefits**:
- Regularization (reduces overfitting)
- Encourages feature diversity
- More effective than standard dropout for deep networks

**Typical**: `drop_rate = 0.2-0.3`

---

### 7. Skip Connections

**Two modes supported**:

**a) Residual (Addition)** - ResNet style:
```python
output = input + conv_block(input)
```

**b) Dense (Concatenation)** - DenseNet style:
```python
output = concat([input, conv_block(input)])
output = 1×1_conv(output)  # Project back to out_channels
```

**DIC uses**: Concatenation (`connection='concatenate'`)
- Richer feature aggregation
- Better gradient flow
- Slightly more parameters (due to 1×1 projection)

---

### 8. Compound Scaling

**Strategy**: Balance width (channels) and depth (layers)

**Formula**:
```python
factor = 2  # Target: 2× parameters
factor_repeat = 1.25  # Depth multiplier
factor_Ns = sqrt(2 / factor_repeat) = 1.26  # Width multiplier

# Ensures: params(scaled) ≈ factor² × params(base)
```

**Rationale**:
- Scaling only depth or width is suboptimal
- Balanced scaling (EfficientNet principle) is most efficient
- Allows systematic model size exploration

---

## DIC Network Configuration

### Best Configuration (324k parameters)

**From notebook** (`training-4.ipynb`, cell 12):

```python
# Input processing
N0_sub_channels = 7   # Shared conv intermediate channels
N0 = 14               # Shared conv output channels

# Encoder (4 blocks)
Ns_0 = [16, 24, 32, 64]          # Channels per block
repeats_0 = [2, 2, 3, 3]          # Layers per block
expansions = [3, 4, 4, 4]         # Expansion ratios
nFilterPerMap = [1, 1, 1, 1]      # Depthwise multiplier
poolings = [
    None,                         # No pooling after block 0
    {'which': 'average', 'stride': 2, 'kernel_size': 2},  # Pool after block 1
    {'which': 'average', 'stride': 2, 'kernel_size': 2},  # Pool after block 2
    None                          # No pooling after block 3
]

# Decoder (for DIC regression task - not needed for classification)
Ns_upsample = [30, 40, 50]       # Upsampling channels
decode_groups = 2                 # Grouped convolutions

# Architecture details
activation = 'Mish'
norm = {'which': 'group', 'num_groups': 1}
weight_scaling = 'normalization'
connection = 'concatenate'
drop_connect_rate = 0.0  # Not used in best config
```

### Training Configuration

**Learning rate schedule**:
```python
def scheduler_func(epoch, vmin=1e-5, vmax=1e-4, cste=0.94, N2=1500):
    if epoch < N2:
        return vmin + (vmax - vmin) * cste**epoch
    return 0.1 * (vmin + (vmax - vmin) * cste**(epoch - N2))
```

**Characteristics**:
- Exponential decay from `vmax` to `vmin`
- Plateau at epoch `N2`, then drop by 10×
- Very gradual decay (cste=0.94 is close to 1)

**Optimizer**: Adam (default PyTorch settings)

---

## Adaptation for Audio Tagging

### Proposed Architecture

**Target**: 300-500k parameters for 80-class audio tagging

```python
# Encoder configuration
in_channels = 1  # Mono spectrogram
encoder_channels = [24, 32, 48, 96]  # 4 blocks
encoder_repeats = [2, 2, 3, 3]        # Total: 10 SepConv layers
expansions = [4, 4, 6, 6]             # Higher for deeper layers

# Pooling strategy (aggressive for audio)
poolings = [
    {'which': 'average', 'stride': 2, 'kernel_size': 2},  # After block 0: 128→64 freq
    {'which': 'average', 'stride': 2, 'kernel_size': 2},  # After block 1: 64→32 freq
    {'which': 'average', 'stride': 2, 'kernel_size': 2},  # After block 2: 32→16 freq
    None  # After block 3: keep 16 freq bins
]

# Global pooling + classifier
global_pool = 'adaptive_avg'  # (B, 96, 16, T) → (B, 96, 1, 1)
classifier = 'linear'          # (B, 96) → (B, 80)

# Architecture details
activation = 'Mish'
norm = {'which': 'group', 'num_groups': 1}
weight_scaling = 'normalization'
se_reduction = 4
drop_connect_rate = 0.2
connection = 'concatenate'
```

### Key Differences from DIC

1. **No decoder**: Classification vs regression
2. **More aggressive pooling**: Reduce time/freq dimensions
3. **Higher expansion ratios**: Audio needs more feature capacity
4. **Global pooling**: Handles variable-length inputs
5. **DropConnect enabled**: Regularization for limited data

---

## Parameter Efficiency Comparison

| Architecture | Parameters | Accuracy (DIC) | Precision/Param Ratio |
|-------------|-----------|----------------|----------------------|
| DIC Baseline (fewer blocks) | 225k | Lower | Good |
| **DIC Best** (v0 config) | **324k** | **High** | **Excellent** |
| DIC Scaled (factor=2) | 1.88M | Higher (diminishing returns) | Lower |

**Key insight**: 300-500k parameter range is sweet spot for efficiency

---

## Training Insights from Notebook

### Iterative Refinement

**DIC uses iterative prediction**: First prediction → warp image → second prediction

**Application to audio**: Not directly applicable (audio is feed-forward)

**Lesson**: Multi-scale processing can help (use different resolutions)

---

### Weight Scaling Application

**From notebook** (cell 12):
```python
trainer.set_optimizer(optimizer_info)
trainer.set_scheduler(scheduler_info)

# Weight scaling called periodically during training
if apply_weight_scaling:
    model.apply_weight_scaling()
```

**Recommendation**: Call after each optimizer step or every N steps

---

### Manual Seed for Reproducibility

**DIC sets seeds explicitly**:
```python
i_seed = 0
torch.manual_seed(i_seed)
```

**Ensures**: Reproducible results across runs

---

## Implementation Checklist

For audio tagging implementation:

### Core Components

- [ ] `WeightNormalizedConv2d` - Weight normalization
- [ ] `SqueezeAndExcitation` - Channel attention
- [ ] `DropConnect` - Stochastic depth
- [ ] `SepConvLayer` - Separable convolution block
- [ ] `SepConvBlock` - Stack of SepConvLayers
- [ ] `get_norm2d()` - Normalization factory (supports GroupNorm)

### Model Architecture

- [ ] `EfficientAudioCNN` - Main model class
- [ ] Encoder blocks with pooling
- [ ] Global average pooling
- [ ] Linear classifier head
- [ ] `apply_weight_scaling()` method

### Training

- [ ] Custom learning rate scheduler (exponential decay)
- [ ] Weight normalization in training loop
- [ ] Mixed precision (AMP) for M4 GPU
- [ ] Reproducible seeds

### Testing

- [ ] Forward pass test
- [ ] Parameter count test
- [ ] Weight normalization test
- [ ] SE block functionality test
- [ ] DropConnect train/eval mode test

---

## Estimated Parameter Count

**For proposed audio config**:

```
Rough calculation:
- Block 0: 2 × SepConvLayer(1→24, expansion=4) ≈ 5k params
- Block 1: 2 × SepConvLayer(24→32, expansion=4) ≈ 15k params
- Block 2: 3 × SepConvLayer(32→48, expansion=6) ≈ 40k params
- Block 3: 3 × SepConvLayer(48→96, expansion=6) ≈ 120k params
- Classifier: 96 × 80 = 7.7k params
- SE blocks: ~10k params total

Total: ~200-250k parameters (base)

With concatenation connections: ~300-400k parameters
With additional features: ~400-500k parameters

Target range: ✅ 300-500k
```

---

## Key Takeaways

1. **Weight normalization** is critical for training stability
2. **Mish activation** significantly outperforms ReLU
3. **Group normalization** (layer norm) better than BatchNorm for spectrograms
4. **Squeeze-and-Excitation** adds minimal overhead with significant gains
5. **Separable convolutions** drastically reduce parameters while maintaining capacity
6. **DropConnect** provides effective regularization
7. **Compound scaling** enables systematic architecture exploration
8. **300-500k parameters** is optimal efficiency range

---

## Next Steps

1. Implement all components in `src/models/components.py`
2. Build `EfficientAudioCNN` in `src/models/efficient_cnn.py`
3. Write comprehensive tests
4. Compare to baseline AudioCNN
5. Train and evaluate performance

---

**Document Status**: ✅ Complete
**Last Updated**: 2026-04-01