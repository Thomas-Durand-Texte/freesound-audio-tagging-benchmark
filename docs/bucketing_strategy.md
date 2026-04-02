# Audio Duration Bucketing Strategy

## Overview

This document describes the recommended bucketing strategy for handling variable-length audio files in the FSDKaggle2019 dataset. The strategy is based on the statistical analysis documented in `reports/dataset_stats.json`.

## Dataset Characteristics

### Curated Set
- **Files**: 4,970
- **Duration statistics**:
  - Mean: 7.65s, Median: 4.68s, Std: 7.70s
  - Range: 0.30s to 57.57s
  - Percentiles: P25=1.61s, P75=11.15s, P95=24.50s

### Noisy Set
- **Files**: 19,815
- **Duration distribution**: Highly concentrated at 15s (75% exactly 15s)
  - Mean: 14.59s, Median: 15.00s, Std: 1.62s
  - Range: 1.05s to 16.00s

## Bucketing Strategy

### Curated Set: Duration-Based Bucketing

Use 5 buckets to balance computational efficiency and memory usage:

| Bucket | Duration Range | Batch Size | Coverage |
|--------|---------------|------------|----------|
| 1      | 0-3s          | 64         | ~20%     |
| 2      | 3-7s          | 48         | ~20%     |
| 3      | 7-15s         | 32         | ~20%     |
| 4      | 15-30s        | 16         | ~20%     |
| 5      | 30s+          | 8          | ~20%     |

**Rationale**:
- Larger batch sizes for shorter clips maximizes GPU utilization
- Smaller batch sizes for longer clips prevents OOM errors
- Approximately equal coverage per bucket ensures balanced training
- Bucket boundaries align with natural breakpoints in duration distribution

### Noisy Set: Fixed Duration

**Single configuration**:
- Duration: 15s (matches 75% of samples exactly)
- Batch size: 32
- Short samples (<15s): pad with zeros
- Long samples (rare): crop to 15s

**Rationale**:
- Minimal preprocessing needed for majority of samples
- Simplified training pipeline
- Consistent memory footprint

## Implementation Notes

1. **Dynamic batching**: Group samples by bucket during data loading
2. **Padding strategy**: Zero-pad to bucket maximum within each batch
3. **Memory optimization**: Pre-allocate tensors based on bucket size
4. **Training**: Iterate through buckets sequentially or with stratified sampling

## References

- Dataset statistics: `reports/dataset_stats.json`
- Decision record: `.blueprints/decisions/D006_variable_length_handling.md`