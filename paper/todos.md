## minor todos

- [x] improve the code to allow early stopping and save the best model
- 

- [ ] exp3 re-weighting baseline
  - [ ] Run Experiments: Start with ./scripts/run_psnet12_cb_loss_exp.sh
  - [ ] Monitor Training: Check logs for class distribution and loss values
  - [ ] Compare Results: Use the comparison script to evaluate CB loss vs baseline
  - [ ] Tune Hyperparameters: Adjust β, γ, α based on your dataset's class imbalance



## justification of the CB loss

### Step-by-Step Logic of the calculate_sample_per_class function

1. **Initialize counters**: `samples_per_class = [0] * 12` for PSNet12

2. **Iterate through sub-clouds**: Each sub-cloud is a processed point cloud region
   ```python
   for labels in dataset.sub_clouds_points_labels:  # e.g., 100 sub-clouds
   ```

3. **Count points per class in each sub-cloud**:
   ```python
   unique, counts = np.unique(labels, return_counts=True)
   # Example: unique=[0,1,4,7], counts=[1500,800,200,50]
   ```

4. **Accumulate across all sub-clouds**:
   ```python
   samples_per_class[0] += 1500  # pipe points
   samples_per_class[1] += 800   # elbow points  
   samples_per_class[4] += 200   # pump points
   samples_per_class[7] += 50    # reducer points
   ```

### Is This Meaningful for PSNet12?

**YES, absolutely!** Here's why:

1. **Point-Level Imbalance**: Even if you have equal numbers of pipe and valve objects, pipes are typically larger → more points per pipe
2. **Training Dynamics**: The loss function operates on individual points, so point-level statistics matter
3. **Real Impact**: A class with 100x fewer points will be overwhelmed during training


### A. Why Point-Level Statistics Matter More Than Object Counts

**Gradient Contribution**: In point cloud semantic segmentation:
```python
# Each point contributes independently to the loss
loss = 0
for each_point in point_cloud:
    loss += cross_entropy(prediction[point], ground_truth[point])
```

If "strainer" has 500 points vs "pipe" with 35,000 points, the strainer class contributes **70x less** to gradient updates!

### B. Effective Number Formula Deep Dive

```python
effective_num = (1 - β^n) / (1 - β)
```

**Why this works**:
- **Small n** (tail classes): `effective_num ≈ n` (linear relationship)
- **Large n** (head classes): `effective_num` saturates due to `β^n → 0`

**Intuition**: Having 1000 vs 2000 samples matters more than having 30,000 vs 31,000 samples.

### C. β Parameter Choice

```python
# Our default: β = 0.9999
cb_beta: 0.9999      # For long-tailed distributions
cb_beta: 0.99999     # For extremely long-tailed (strong variant)
```

**Rule of thumb**: Higher β = more aggressive re-weighting for tail classes.

### D. Integration with Focal Loss

The CB loss combines two orthogonal benefits:
1. **Class-level rebalancing** (CB weighting)
2. **Instance-level hard example mining** (focal loss γ parameter)

```python
final_loss = cb_weight × (1 - p_t)^γ × cross_entropy
#            ↑class      ↑hard examples  ↑base loss
```

### E. Masking Compatibility

Critical for point clouds where not all points are valid:
```python
# Only count valid points toward class statistics
valid_idx = mask_flat > 0
logits_valid = logits_flat[valid_idx]
targets_valid = targets_flat[valid_idx]
```

This ensures CB loss works correctly with:
- Variable point cloud sizes
- Padded batches
- Sparse annotations

## Summary: Why CB Loss is Perfect for PSNet12

1. **Real Imbalance**: 70x difference between most/least common classes
2. **Point-Level Training**: Loss computed per-point, so point statistics matter
3. **Gradient Fairness**: Without CB loss, tail classes get overwhelmed
4. **Practical Impact**: Tail classes (valves, strainers) are often most important for industrial applications!

The `calculate_samples_per_class` function is absolutely meaningful because it captures the **actual training signal imbalance** that the model experiences during learning.