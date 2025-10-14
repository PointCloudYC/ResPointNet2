# PosPool and PseudoGrid Models for PSNet12

This document describes the newly added PosPool and PseudoGrid (KPConv) models for benchmarking on the PSNet12 dataset.

## Overview

We have successfully integrated two additional local aggregation operators from CloserLook3D:
- **PosPool**: Position-based pooling with learnable position embeddings
- **PseudoGrid (KPConv)**: Kernel Point Convolution using pseudo-grid sampling

Both models are optimized for the A6000 GPU setup.

## Model Architectures

### PosPool
Position-based pooling operator that learns spatial relationships through position embeddings. Two variants:
- **xyz encoding**: Direct 3D coordinate embedding
- **sin_cos encoding**: Sinusoidal-cosine position encoding (similar to Transformer positional encoding)

### PseudoGrid (KPConv)
Kernel Point Convolution that uses learnable kernel points positioned in a pseudo-grid around each query point.
- Uses 15 kernel points by default
- Linear influence function for kernel weighting
- More expressive than standard PointNet++ operators

## Configuration Files

All configs are located in `cfgs/psnet12/`:

### PosPool Configurations
1. **pospool_xyz_avg_a6000.yaml** - PosPool with XYZ position embedding
2. **pospool_sin_cos_avg_a6000.yaml** - PosPool with sin-cos position embedding

### PseudoGrid Configuration
1. **pseudo_grid_a6000.yaml** - PseudoGrid (KPConv) configuration

## Training Scripts

Training scripts are located in `scripts/`:

### PosPool Training
```bash
# Train PosPool with XYZ encoding
bash scripts/run_psnet12_pospool_a6000.sh [patience]

# Example with custom patience
bash scripts/run_psnet12_pospool_a6000.sh 50
```

### PseudoGrid Training
```bash
# Train PseudoGrid (KPConv)
bash scripts/run_psnet12_pseudogrid_a6000.sh [patience]

# Example with custom patience
bash scripts/run_psnet12_pseudogrid_a6000.sh 50
```

## A6000 Optimizations

All configurations include the following optimizations for 2x A6000 GPUs:

1. **Increased Batch Size**: 16 (up from 6)
   - Utilizes 48GB VRAM per GPU effectively
   - Approximately 2.67x increase

2. **Scaled Learning Rate**: 0.0267 (up from 0.01)
   - Linear scaling rule: new_lr = base_lr × (new_batch / old_batch)

3. **Increased Workers**: 8 (up from 4)
   - Better data loading throughput

4. **Warmup Epochs**: 5
   - Helps stabilize training with larger batch size

5. **Mixed Precision**: O1 level
   - ~2x speedup on A6000
   - Reduces memory usage

## Key Configuration Parameters

### PosPool Parameters
```yaml
local_aggregation_type: 'pospool'
pospool:
  position_embedding: 'xyz'  # or 'sin_cos'
  reduction: 'avg'           # max, avg, or sum
  output_conv: False
```

### PseudoGrid Parameters
```yaml
local_aggregation_type: 'pseudo_grid'
pseudo_grid:
  fixed_kernel_points: 'center'  # center or verticals
  KP_influence: 'linear'         # constant, linear, or gaussian
  KP_extent: 1.0
  num_kernel_points: 15
  convolution_mode: 'sum'
  output_conv: False
density_parameter: 5.0  # Required for kernel extent calculation
```

## Expected Performance

Training times (estimated on 2x A6000):
- **PosPool**: ~24-36 hours for 600 epochs
- **PseudoGrid**: ~30-42 hours for 600 epochs (slightly slower due to kernel point computations)

Early stopping with patience=50 may reduce training time significantly if the model converges early.

## Evaluation

Both training scripts automatically run evaluation after training completes, using the best checkpoint (based on validation mIoU).

Evaluation results are logged in the same directory as training logs:
- `log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/`
- `log/psnet12/pseudo_grid_a6000_TIMESTAMP/`

## Implementation Details

### Code Changes

1. **models/local_aggregation_operators.py**
   - Added `PosPool` class
   - Added `PseudoGrid` class
   - Updated `LocalAggregation` to support both new operators

2. **utils/config.py**
   - Added default config structures for `pospool` and `pseudo_grid`

3. **models/utlis.py** (already existed)
   - Contains required utility functions: `create_kernel_points`, `radius_gaussian`, `weight_variable`

### Dependencies

All dependencies are already present in the existing ResPointNet2 environment:
- PyTorch
- CUDA custom operators (already compiled)
- Standard Python libraries (numpy, etc.)

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce `batch_size` in the config file (try 12 or 14 instead of 16)

### Issue: Kernel points file not found
**Solution**: The first run will create kernel point files in `kernels/dispositions/`. This is normal and only happens once.

### Issue: Training is slow
**Solution**: 
1. Verify `amp_opt_level: 'O1'` is set in config
2. Check that `num_workers: 8` is set
3. Ensure no other processes are using the GPUs

### Issue: Config key errors
**Solution**: Make sure you're using the latest version of `utils/config.py` which includes the new config structures.

## Comparison with ResPointNet++

Expected relative performance (accuracy):
- **ResPointNet++**: Baseline
- **PosPool**: Similar or slightly better (especially with sin_cos encoding)
- **PseudoGrid**: Generally better due to more expressive kernel point convolutions

Training speed (relative):
- **ResPointNet++**: Fastest
- **PosPool**: Slightly slower (~10-15% slower)
- **PseudoGrid**: Slower (~20-30% slower due to kernel computations)


## Future Work

Potential extensions:
1. Add more position encoding variants for PosPool
2. Experiment with different kernel point numbers for PseudoGrid
3. Try adaptive kernel point selection
4. Implement RandLA-Net and other operators from CloserLook3D