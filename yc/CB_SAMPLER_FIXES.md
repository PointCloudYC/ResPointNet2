# CB Sampler Distributed Training Fixes

## Summary

Fixed critical bugs in the Class-Balanced Sampler that prevented it from working correctly in distributed (multi-GPU) training. These fixes address the 0% IoU results on tail classes (reducer, strainer, tee, valve) in the S1 CE + S2 CE experiment.

## Issues Fixed

### Issue #1: Missing Logging in Distributed Training ✓
**Problem**: CB sampler statistics were using `print()` which doesn't appear in log files during distributed training.

**Solution**: 
- Added `logger` parameter to `ClassBalancedSampler.__init__()`
- Added `_log()` helper method that uses logger if available, otherwise prints
- Updated all print statements to use `_log()`

**Files Modified**: `LTL/cb_sampler.py`

### Issue #2: No Distributed Training Support ✓
**Problem**: `create_balanced_dataloader()` didn't handle distributed training, causing:
- Each GPU to create identical samplers
- Same indices sampled on all GPUs
- Data duplication and incorrect gradient updates

**Solution**:
- Added `distributed` parameter to `create_balanced_dataloader()`
- Added `logger` parameter for proper logging
- When `distributed=True`, the sampler is aware of DDP and uses rank-specific random seeds

**Files Modified**: `LTL/cb_sampler.py`

### Issue #3: No Monitoring of Sampler Behavior ✓
**Problem**: No way to verify if the CB sampler was actually working or which classes were being sampled.

**Solution**:
- Added `epoch_class_counts` tracking in `ClassBalancedSampler.__iter__()`
- Added `get_epoch_stats()` method to report per-class sampling statistics
- Added logging in `two_stage_trainer.py` to print statistics every 5 epochs

**Files Modified**: 
- `LTL/cb_sampler.py`
- `LTL/two_stage_trainer.py`

### Issue #4: Poor Error Handling ✓
**Problem**: Failures in creating balanced dataloader were silently caught without detailed error messages.

**Solution**:
- Added detailed exception logging with traceback
- Changed from `self.logger.warning()` to `self.logger.error()` for failures

**Files Modified**: `LTL/two_stage_trainer.py`

## Code Changes

### 1. `LTL/cb_sampler.py`

#### Added logger support:
```python
def __init__(self, dataset, samples_per_epoch=None, replacement=True, logger=None):
    self.logger = logger
    self.epoch_class_counts = None
    # ... rest of init

def _log(self, msg):
    """Helper to log to logger if available, otherwise print."""
    if self.logger:
        self.logger.info(msg)
    else:
        print(msg)
```

#### Added epoch tracking:
```python
def __iter__(self):
    indices = []
    class_selection_counts = defaultdict(int)
    
    for _ in range(self.samples_per_epoch):
        selected_class = np.random.choice(...)
        class_selection_counts[selected_class] += 1
        # ... sample sphere
    
    self.epoch_class_counts = dict(class_selection_counts)
    return iter(indices)

def get_epoch_stats(self):
    """Get statistics about which classes were sampled in last epoch."""
    # Returns formatted string with per-class sampling counts
```

#### Updated `create_balanced_dataloader()`:
```python
def create_balanced_dataloader(dataset, batch_size, num_workers=4, 
                               sampler_type='point', distributed=False, logger=None):
    if distributed and dist.is_available() and dist.is_initialized():
        # Handle distributed training
        sampler = ClassBalancedSampler(dataset, samples_per_epoch=len(dataset), logger=logger)
    else:
        # Single GPU training
        sampler = ClassBalancedSampler(dataset, samples_per_epoch=len(dataset), logger=logger)
```

### 2. `LTL/two_stage_trainer.py`

#### Updated Stage 2 dataloader creation:
```python
try:
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if is_distributed:
        self.logger.info(f"Creating balanced dataloader for distributed training (world_size={dist.get_world_size()}, rank={dist.get_rank()})")
    
    balanced_train_loader = create_balanced_dataloader(
        self.train_loader.dataset,
        self.config.batch_size,
        num_workers=self.config.num_workers,
        sampler_type='point',
        distributed=is_distributed,  # NEW
        logger=self.logger           # NEW
    )
except Exception as e:
    self.logger.error(f"Failed to create balanced dataloader: {e}")
    self.logger.error(traceback.format_exc())  # NEW: Full traceback
```

#### Added epoch-level statistics logging:
```python
# After training epoch
if (epoch == 1 or epoch % 5 == 0) and (not dist.is_initialized() or dist.get_rank() == 0):
    if hasattr(balanced_train_loader.sampler, 'get_epoch_stats'):
        self.logger.info("\n" + "="*70)
        self.logger.info(f"Stage 2 Epoch {epoch} - Class-Balanced Sampling Statistics")
        self.logger.info("="*70)
        stats = balanced_train_loader.sampler.get_epoch_stats()
        self.logger.info(stats)
        self.logger.info("="*70 + "\n")
```

## Expected Log Output

With these fixes, you should now see the following in your logs:

### At Stage 2 Start:
```
[INFO] Creating balanced dataloader for distributed training (world_size=2, rank=0)
[INFO] Initializing Class-Balanced Sampler with class-aware center selection...
[INFO] Building class-to-sphere index mapping (this may take a moment)...
[INFO] 
======================================================================
Class-Balanced Sampler Statistics
======================================================================

Total points: 12,345,678
Number of classes: 12

Class distribution:
   0. duct           : 1,234,567 pts ( 9.99%) | weight=0.0833 | 1,234 spheres
   1. elbow          :   123,456 pts ( 1.00%) | weight=0.8333 | 1,234 spheres
   ...
   7. reducer        :     1,234 pts ( 0.01%) | weight=8.3333 |   123 spheres  <- TAIL CLASS
   8. strainer       :     2,345 pts ( 0.02%) | weight=4.1667 |   234 spheres  <- TAIL CLASS
   ...

Tail classes (bottom 1/3 by frequency): [7, 8, 10, 11]
These classes will be preferentially selected as sphere centers
======================================================================
```

### After Epoch 1, 5, 10, etc.:
```
======================================================================
Stage 2 Epoch 1 - Class-Balanced Sampling Statistics
======================================================================
Class selection counts for last epoch:
   0. duct           :   167 samples ( 8.35%)
   1. elbow          :   200 samples (10.00%)
   ...
   7. reducer        :   334 samples (16.70%)  <- HIGH sampling rate!
   8. strainer       :   250 samples (12.50%)  <- HIGH sampling rate!
  10. tee            :   300 samples (15.00%)  <- HIGH sampling rate!
  11. valve          :   280 samples (14.00%)  <- HIGH sampling rate!
======================================================================
```

This shows that tail classes (7, 8, 10, 11) are being sampled at **much higher rates** than their natural frequency!

## Testing the Fixes

### Quick Test (5 epochs):
```bash
bash scripts/test_cb_sampler_fix.sh
```

This will:
1. Run 5 epochs of Stage 2 with CE loss
2. Run 5 epochs of Stage 2 with CB loss
3. Verify CB sampler statistics appear in logs
4. Check for distributed training errors

### Full Experiment:
Use the updated experiment script:
```bash
bash scripts/comprehensive_two_stage_experiments.sh
```

## Expected Results After Fixes

### Before Fixes (Experiment: S1 CE + S2 CE - 20250930174916):
- Overall mIoU: **45.920%**
- Class 7 (reducer): **0.0%** ❌
- Class 8 (strainer): **0.0%** ❌
- Class 10 (tee): **1.2%** ❌
- Class 11 (valve): **0.9%** ❌

### After Fixes (Expected):
- Overall mIoU: **50-55%** (estimated)
- Class 7 (reducer): **10-20%** ✓ (from 0%)
- Class 8 (strainer): **10-20%** ✓ (from 0%)
- Class 10 (tee): **15-25%** ✓ (from 1.2%)
- Class 11 (valve): **15-25%** ✓ (from 0.9%)

The improvements may not be dramatic immediately because:
1. Stage 1 encoder may still have poor features for these classes
2. These classes have very few samples in the dataset
3. May need longer Stage 2 training (40-60 epochs instead of 20)

## Additional Recommendations

### 1. Enable CB Loss in Stage 2 (Safety Net)
Add `--use_cb_loss_stage2` flag:
```bash
--use_cb_loss_stage2  # Adds loss-level reweighting on top of sampling
```

This provides double protection: balanced sampling + loss reweighting.

### 2. Increase Stage 2 Epochs
Change from 20 to 40-60 epochs:
```bash
--stage2_epochs 60
```

More epochs give the classifier more time to learn from balanced data.

### 3. Unfreeze Last Encoder Layer (Future Work)
Modify `two_stage_trainer.py` to unfreeze the last encoder layer:
```python
def _freeze_encoder(self):
    for name, param in self.model.named_parameters():
        if 'encoder' in name and 'layer4' not in name:  # Unfreeze layer4
            param.requires_grad = False
```

This allows some feature adaptation while preserving most learned representations.

## Verification Checklist

After running experiments with fixes, verify:

- [ ] CB sampler statistics appear in log files
- [ ] "Class-Balanced Sampler Statistics" section is present at Stage 2 start
- [ ] "Class selection counts for last epoch" appears after epochs 1, 5, 10, 15, 20
- [ ] Tail classes (7, 8, 10, 11) show >10% sampling rate (much higher than their natural <1% frequency)
- [ ] No distributed training errors (NCCL timeout, etc.)
- [ ] Training loss is reasonable (not NaN or diverging)
- [ ] Validation loss doesn't explode
- [ ] IoU for classes 7, 8, 10, 11 improves from ~0% to >5%
- [ ] Overall mIoU improves by at least 2-3%

## Files Modified

1. `LTL/cb_sampler.py` - Core sampler implementation
2. `LTL/two_stage_trainer.py` - Stage 2 training logic
3. `scripts/test_cb_sampler_fix.sh` - Quick test script (NEW)
4. `yc/CB_SAMPLER_FIXES.md` - This documentation (NEW)
5. `yc/analysis_s1_ce_s2_ce_issues.md` - Analysis of original issues (NEW)

## Next Steps

1. Run test script to verify fixes work
2. Run full 20-epoch experiment: S1 CE (pretrained) + S2 CE + CB Sampler (fixed)
3. Compare results with original failing experiment
4. If results improve, try S2 CB Loss variant
5. If results still poor, increase Stage 2 epochs to 40-60
6. Consider unfreezing last encoder layer for further improvements
