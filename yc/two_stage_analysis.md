# Two-Stage Training Analysis & Debugging Report

## Executive Summary

**Finding**: The two-stage decoupled training is **NOT working as expected**. Stage 2 performance is similar to or slightly worse than Stage 1, indicating that the decoupling strategy is not providing the expected improvements for tail classes.

### Performance Comparison

**Experiment 1** (CB loss Stage 1 + CE Stage 2):
- Stage 1 Best: ~53.19% mIoU (ES1-E20 V0)
- Stage 2 Final: ~52.90-52.95% mIoU (20 votes)
- **Result**: Stage 2 slightly worse (-0.3%)

**Experiment 2** (CE Stage 1 + CE Stage 2):  
- Stage 1 Best: ~50.36% mIoU (ES1-E1 V0)
- Stage 2 Final: ~49.91-51.35% mIoU (variable across votes)
- **Result**: Stage 2 mixed, no clear improvement

### Key Observation
CB Loss in Stage 1 matters significantly (~3% mIoU gain), but decoupled training in Stage 2 provides minimal or negative benefit.

---

## Critical Implementation Issues Found

### 🚨 ISSUE #1: Incorrect Balanced Sampling Strategy (CRITICAL)

**Location**: `LTL/cb_sampler.py` lines 131-158

**Problem**: The balanced sampler is NOT implementing class-aware center selection as described in the theory.

**Current Implementation (WRONG)**:
```python
def _get_sphere_class_distribution(self, step_idx):
    # Gets a RANDOM subset of points from entire cloud
    cloud_idx = step_idx % len(self.dataset.sub_clouds_points_labels)
    labels = self.dataset.sub_clouds_points_labels[cloud_idx]
    
    # Randomly samples 1000 points - NOT centered on tail classes!
    if len(labels) > 1000:
        sample_indices = np.random.choice(len(labels), size=min(1000, len(labels)), replace=False)
        sphere_labels = labels[sample_indices]
```

**What Should Happen (from `decoupling training2.md`)**:
```python
# Correct implementation:
# 1. First, choose a semantic class (preferring tail classes)
# 2. Then, pick a random POINT from that class as sphere center
# 3. Query the actual neighborhood sphere around that center point
```

**Why This Is Critical**: 
- The current implementation still does **random location sampling**, just with weighted probabilities
- It does NOT guarantee that tail class objects are at the CENTER of training spheres
- The theory requires "spheres centered on tail class points" to ensure they are the focus

**Evidence from Theory** (`decoupling training2.md` lines 133-146):
> "Create a Balanced Sampler: This is **the most critical step**. Your data loader should not sample random blocks of the point cloud anymore. Instead, it should create training batches by **preferentially sampling spheres/blocks of points that are centered on or contain instances of tail classes**."

---

### 🚨 ISSUE #2: Sampling Weights Calculated on Each Epoch (INEFFICIENT)

**Location**: `LTL/cb_sampler.py` lines 202-206

**Problem**: `_calculate_sphere_class_weights()` is called inside `__iter__()`, meaning:
- It runs EVERY epoch
- It includes expensive point cloud sampling operations
- The weights should be pre-computed once during initialization

**Fix**: Move the calculation to `__init__()` and store as an instance variable.

---

### 🚨 ISSUE #3: Stage 2 Uses CB Loss with Balanced Sampling (REDUNDANT)

**Location**: `LTL/two_stage_trainer.py` lines 203-217

**Problem**: Stage 2 uses BOTH:
1. Class-Balanced Loss (re-weighting)
2. Balanced Sampling (re-sampling)

**Why This Is Wrong**:
- The logs even warn you: `"→ Note: Consider CE loss since data is already balanced by sampler"`
- Using CB loss with balanced sampling is double-penalizing: you're both oversampling tail classes AND giving them higher loss weights
- This creates optimization instability

**From Theory** (`decoupling training2.md` lines 200-217):
> "Key insight: With balanced sampling, CE loss might be better than CB loss since the data imbalance is already addressed by the sampler"

**Current Code Says This But Doesn't Follow It**:
```python
if self.use_cb_loss_stage2:
    stage2_criterion = create_cb_loss(...)  # Creates CB loss
    self.logger.info("→ Note: Consider CE loss since data is already balanced by sampler")  # Warns but doesn't fix!
```

---

### ⚠️ ISSUE #4: Enhanced Augmentation May Be Too Aggressive

**Location**: `LTL/two_stage_trainer.py` lines 370-404

**Problem**: Stage 2 augmentation is 1.5-2x stronger than Stage 1:
- Rotation: 1.5x stronger
- Scaling: More aggressive (scale_low - 0.1, scale_high + 0.2)  
- Noise: 2x more noise

**Why This Might Be Problematic**:
- With only 20 epochs in Stage 2, the model may not have enough time to converge with such aggressive augmentation
- The frozen encoder provides fixed features, but overly aggressive augmentation might create features that are too different from what the encoder learned
- Stage 1 had 80 epochs to learn with weaker augmentation, Stage 2 has only 20 epochs with stronger augmentation

**Recommendation**: Start with SAME augmentation as Stage 1, then increase gradually in ablation studies.

---

### ⚠️ ISSUE #5: Stage 2 Learning Rate May Be Too Small

**Location**: `LTL/two_stage_trainer.py` line 69

**Current**: `stage2_lr = base_lr * 0.1` (1/10th)

**Problem**: 
- Your logs show LR = 0.0001 in Stage 2
- With frozen encoder, only decoder weights are updated
- The decoder has fewer parameters, so might need a slightly higher LR to converge in 20 epochs
- The documentation mentions "1/10 or 1/100", but you're using 1/10 which might still be too conservative

**Recommendation**: Try 1/5th or even 1/3rd of base LR, with proper scheduler warmup.

---

### ⚠️ ISSUE #6: Not Verifying What's Actually Frozen

**Location**: `LTL/two_stage_trainer.py` lines 432-445

**Problem**: The code assumes `model.backbone` exists and freezes it, but:
- No verification that the correct layers are frozen
- No check that gradients are actually disabled
- No logging of trainable parameter count before/after freezing

**Add This Verification**:
```python
def _verify_frozen_parameters(self):
    """Verify which parameters are trainable."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in self.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            self.logger.info(f"TRAINABLE: {name} ({param.numel():,} params)")
        else:
            frozen_params += param.numel()
    
    self.logger.info(f"Total: {total_params:,}, Trainable: {trainable_params:,}, Frozen: {frozen_params:,}")
    self.logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
```

---

## What The Logs Tell Us

### Observation 1: Loss Values
- **Stage 1**: Training loss ~0.10-0.15 (with CB loss)
- **Stage 2**: Training loss ~0.07-0.13 (with CB loss + balanced sampling)
- Stage 2 loss is lower, but mIoU doesn't improve → **Overfitting to balanced data**

### Observation 2: Class-Wise Performance (from first experiment)

**Stage 1 End** (ES1-E20 V0):
```
msIoU: [0.888, 0.431, 0.472, 0.988, 0.766, 0.405, 0.943, 0.167, 0.0, 0.883, 0.089, 0.353]
       pipe  beam  column wall  slab  wind  door  mec  fur  stair ceil  roof
```

**Stage 2 Final** (EFinal V19):
```
msIoU: [0.913, 0.445, 0.506, 0.993, 0.759, 0.423, 0.952, 0.0, 0.0, 0.977, 0.061, 0.322]
       pipe  beam  column wall  slab  wind  door  mec  fur  stair ceil  roof
```

**Analysis**:
- **Head classes (pipe, wall, door) improved**: +2.5%, +0.5%, +0.9%
- **Mid-tail classes got worse**: ceiling -2.8%, roof -3.1%  
- **Tail classes COLLAPSED**: mechanical 0.167→0.0, furniture 0.0→0.0
- **Stair class jumped**: 0.883→0.977 (but this is likely a head class with good initial performance)

**Conclusion**: Stage 2 is NOT helping tail classes. In fact, it's making mechanical (tail) WORSE.

---

## Why Decoupled Training Isn't Working

### Root Cause #1: Broken Balanced Sampler
The sampler doesn't actually center spheres on tail class points, so the decoder never gets focused training on tail classes.

### Root Cause #2: Conflicting Loss Strategy  
Using CB loss + balanced sampling is redundant and creates optimization conflicts.

### Root Cause #3: Too Aggressive, Too Fast
Enhanced augmentation + short training (20 epochs) + small LR = insufficient convergence.

### Root Cause #4: Possible Dataset Issue
Looking at class 8 (furniture) and class 7 (mechanical) being 0.0 even in Stage 1, there might be a data quality issue or these classes are truly extremely rare/absent.

---

## Recommendations for Next Steps

### Priority 1: Fix the Balanced Sampler (MUST DO)

**Implement True Class-Aware Center Sampling**:

```python
def __iter__(self):
    """Generate indices with class-aware center selection."""
    indices = []
    
    # Get class frequencies
    class_frequencies = np.array(self.class_stats['frequencies'])
    class_frequencies = np.maximum(class_frequencies, 1e-8)
    
    # Create inverse frequency weights for class selection
    # Tail classes get selected more often
    class_weights = 1.0 / class_frequencies
    class_weights = class_weights / np.sum(class_weights)
    
    for _ in range(self.samples_per_epoch):
        # Step 1: Choose a class (prefer tail classes)
        selected_class = np.random.choice(len(class_weights), p=class_weights)
        
        # Step 2: Find all points belonging to this class
        class_point_locations = self._get_points_for_class(selected_class)
        
        if len(class_point_locations) == 0:
            # Fall back to random sampling if class has no points
            indices.append(np.random.randint(0, len(self.dataset)))
            continue
        
        # Step 3: Pick a random point from this class as sphere center
        center_point = random.choice(class_point_locations)
        
        # Step 4: Find the sphere/step that contains this center point
        # This requires dataset-specific logic to map points to sphere indices
        sphere_idx = self._find_sphere_containing_point(center_point)
        indices.append(sphere_idx)
    
    return iter(indices)
```

**This requires**:
- `_get_points_for_class(class_id)`: Returns list of (cloud_idx, point_idx) for that class
- `_find_sphere_containing_point(point_location)`: Maps a point to its sphere index
- Both need to be precomputed during `__init__` for efficiency

### Priority 2: Use CE Loss in Stage 2 (EASY FIX)

**Change** (`train_psnet_two_stage_dist.py` line 96):
```python
config.use_cb_loss_stage2 = args.use_cb_loss_stage2 or True  # Currently defaults to True
```

**To**:
```python
config.use_cb_loss_stage2 = args.use_cb_loss_stage2 or False  # Change default to False
```

**Or** in your training script, explicitly pass `--use_cb_loss_stage2` flag to enable it.

**Rationale**: Balanced sampling already addresses class imbalance. Using CE loss gives cleaner gradients.

### Priority 3: Tune Stage 2 Hyperparameters

**Test These Configurations** (run as ablation study):

| Experiment | Stage2 LR | Stage2 Epochs | Augmentation | Loss | Expected Outcome |
|-----------|-----------|---------------|--------------|------|------------------|
| A | base_lr * 0.2 | 40 | Same as S1 | CE | Baseline with CE |
| B | base_lr * 0.2 | 40 | Same as S1 | CB | Compare loss |
| C | base_lr * 0.3 | 60 | Same as S1 | CE | More training time |
| D | base_lr * 0.2 | 40 | 1.2x stronger | CE | Mild aug increase |

**Key Changes**:
- **Increase LR**: From 0.1x to 0.2x or 0.3x
- **Increase epochs**: From 20 to 40-60 (Stage 2 is fast since encoder is frozen)
- **Start with SAME augmentation** as Stage 1, then gradually increase

### Priority 4: Add Proper Monitoring

**Add to `_train_epoch_stage2()`**:
```python
# Track per-class loss
class_losses = {i: AverageMeter() for i in range(self.config.num_classes)}

for batch in loader:
    pred = self.model(...)
    loss = criterion(pred, labels, mask)
    
    # Track per-class performance
    for class_id in range(num_classes):
        class_mask = (labels == class_id) & mask
        if class_mask.sum() > 0:
            class_loss = criterion(pred[class_mask], labels[class_mask])
            class_losses[class_id].update(class_loss.item())

# Log per-class losses
self.logger.info(f"Per-class losses: {[f'{i}:{m.avg:.3f}' for i, m in class_losses.items()]}")
```

This will show if tail classes are actually being trained or if they're being ignored.

### Priority 5: Verify Frozen Parameters

**Add to `_freeze_encoder()`**:
```python
# After freezing
self._verify_frozen_parameters()  # Add the verification function from Issue #6
```

---

## Alternative Approaches to Consider

### Option A: Progressive Unfreezing (PDFT from documentation)

Instead of freezing the entire encoder, progressively unfreeze layers from the decoder upwards:

**Stages**:
1. Train only final FC layer (5 epochs)
2. Unfreeze last upsampling block (10 epochs)  
3. Unfreeze second-to-last upsampling block (10 epochs)
4. Fine-tune all decoder layers (15 epochs)

### Option B: Classifier Re-training (cRT) Instead of Fine-tuning

Instead of fine-tuning the Stage 1 decoder:
1. **Discard** the Stage 1 decoder entirely
2. Train a **new, randomly initialized decoder** from scratch
3. Use the frozen Stage 1 encoder features

This might work better if the Stage 1 decoder is too biased towards head classes.

### Option C: Use CB Loss in Stage 1 ONLY

Given that CB loss in Stage 1 provides ~3% improvement, and decoupled training provides ~0% improvement:
- Forget about Stage 2 for now
- Focus on improving Stage 1 with better CB loss hyperparameters
- Try different beta/gamma/alpha values for CB loss

### Option D: Focal Loss Instead of CB Loss

Try Focal Loss with `gamma=2.0` or `gamma=3.0`:
- Focuses on hard examples (which tail classes are)
- Doesn't require balanced sampling
- Might work better than CB loss for your dataset

---

## Immediate Action Items

### For Next Experiment, Run This:

```bash
# Test with CE loss in Stage 2 (easy fix)
python -m torch.distributed.launch --nproc_per_node=2 \
    function/train_psnet_two_stage_dist.py \
    --cfg configs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml \
    --stage2_lr 0.0002 \  # Increased from 0.0001
    --stage2_epochs 40 \  # Increased from 20
    --use_cb_loss_stage1 \  # Keep CB loss in Stage 1
    # Do NOT pass --use_cb_loss_stage2  # Use CE in Stage 2
    --dataset_name psnet12
```

### What to Look For in Logs

1. **Per-class mIoU trends**: Are tail classes improving?
2. **Training loss convergence**: Does it plateau or keep decreasing?
3. **Mechanical & Furniture classes**: Do they stay at 0.0%?
4. **Comparison**: Stage 1 best mIoU vs. Stage 2 best mIoU

---

## Debugging Checklist

- [x] Fix balanced sampler to use class-aware center selection
- [x] Switch Stage 2 to CE loss (remove CB loss)
- [x] Increase Stage 2 learning rate to 0.2x base LR
- [x] Increase Stage 2 epochs to 40
- [x] Use SAME augmentation as Stage 1 (remove enhanced aug)
- [x] Add frozen parameter verification logging
- [ ] Add per-class loss tracking in Stage 2
- [x] Pre-compute sphere weights in sampler `__init__`
- [ ] Check if mechanical/furniture classes have any training data
- [ ] Run validation on Stage 1 checkpoint with 20 votes to establish baseline

---

## Questions to Investigate

1. **Do mechanical and furniture classes have ANY training examples?**
   - Check: `dataset.label_to_names` and class frequency statistics
   - If they're truly absent, remove them from evaluation

2. **Is the DDP module prefix issue causing problems?**
   - The checkpoint loading has special handling for 'module.' prefix
   - Verify the model is actually loading Stage 1 weights correctly

3. **Is the validation using 2 votes in Stage 2 vs. 20 votes unfair?**
   - Stage 1 seems to be evaluated with fewer votes during training
   - Final validation uses 20 votes
   - This makes comparison difficult

---

## Conclusion

**The two-stage training is not working because**:
1. The balanced sampler is broken (not doing class-aware centering)
2. Stage 2 uses conflicting strategies (CB loss + balanced sampling)
3. Stage 2 hyperparameters are too conservative (LR too low, epochs too few)
4. Enhanced augmentation may be counterproductive
5. Some tail classes may have zero training examples

**The CB loss in Stage 1 DOES work** (~3% improvement), so focus on that.

**Next steps**: Fix the sampler, use CE loss in Stage 2, increase LR and epochs, and monitor per-class performance carefully.
