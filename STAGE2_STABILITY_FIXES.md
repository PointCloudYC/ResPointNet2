# Stage 2 Training Stability Fixes

## Problem Summary

Your Stage 2 training was experiencing **catastrophic performance collapse**, dropping from 52.47% mIoU to 40.56% in just 5 epochs. Analysis of your training log revealed three critical issues that needed to be addressed.

---

## Applied Fixes

### FIX #1: Dramatically Reduced Learning Rate ✅

**Issue:** 
- Learning rate was too aggressive for fine-tuning frozen encoder features
- Performance collapse occurred immediately when warmup ended and full LR (0.0001) was applied
- Evidence: mIoU dropped from 49.7% (epoch 3, warmup) to 40.6% (epoch 5, full LR)

**Solution:**
```python
# BEFORE: stage2_lr = base_lr * 0.2 → effective 0.0001 after 0.5x reduction
# AFTER:  stage2_lr = base_lr * 0.05 → effective 0.000025 (4x smaller!)
```

**Changes Made:**
- `LTL/two_stage_trainer.py` line 75: Changed default from `0.2` to `0.05`
- `function/train_psnet_two_stage_dist.py` line 97: Updated config default
- Effective Stage 2 LR is now **2.5% of Stage 1 LR** (was 10%)

**Expected Impact:**
- Gradual, stable classifier adaptation
- Prevents "catastrophic forgetting" of Stage 1 features
- May need to train for more epochs, but performance will be stable

---

### FIX #2: Extended & More Conservative Warmup ✅

**Issue:**
- 3-epoch warmup was too short for classifier to adapt to:
  1. Frozen encoder features (can't adjust them anymore)
  2. Drastically different class distribution from balanced sampling
- Warmup started at 0.3× base_lr, which was still too aggressive

**Solution:**
```python
# BEFORE: 3 epochs, starting at 0.3× → 1.0× base_lr
# AFTER:  5-10 epochs (auto-scaled), starting at 0.1× → 1.0× base_lr
```

**Changes Made:**
- `LTL/two_stage_trainer.py` lines 340-346: Dynamic warmup calculation
  - Default: `min(10, max(5, stage2_epochs // 2))`
  - For 20-epoch Stage 2: uses 10 epochs of warmup
- `LTL/two_stage_trainer.py` lines 507-519: More conservative ramp
  - Start at 10% instead of 30% of base_lr
  - Gentler slope for adaptation
- Added `--stage2_warmup_epochs` CLI argument for manual override

**Expected Impact:**
- Classifier gets ~2,500 batches to adapt gently (was ~750)
- Reduces shock from distribution shift
- Performance should improve gradually throughout warmup

---

### FIX #3: Temperature Scaling for Balanced Sampling ✅

**Issue:**
- Sampling was **extremely** aggressive:
  - `strainer` (0.10% of data) → 45% of training samples (450× oversampling!)
  - `reducer` (0.12% of data) → 35% of training samples (294× oversampling!)
  - Together: 80% of samples from 0.22% of data
- This created a distribution the frozen encoder had never seen

**Solution:**
```python
# Apply temperature scaling to class weights
# weights = weights^(1/temperature)
# temperature = 2.0 → softens distribution significantly
```

**Changes Made:**
- `LTL/cb_sampler.py` line 28: Added `temperature` parameter to `ClassBalancedSampler`
- `LTL/cb_sampler.py` lines 147-149: Temperature scaling in weight calculation
- `LTL/cb_sampler.py` line 351: Added `temperature` to `create_balanced_dataloader`
- `LTL/two_stage_trainer.py` lines 299-315: Pass temperature to sampler
- `function/train_psnet_two_stage_dist.py` line 54: Added `--sampling_temperature` CLI argument
  - Default: 2.0 (conservative, balanced approach)

**Expected Impact:**
- More balanced distribution between head and tail classes
- Tail classes still boosted, but not to extreme levels
- Encoder features remain aligned with classifier expectations

**Temperature Guidelines:**
- `temperature = 1.0`: Original inverse frequency (very aggressive)
- `temperature = 2.0`: **Recommended default** (good balance)
- `temperature = 3.0`: Very conservative (closer to uniform)
- `temperature < 1.0`: More aggressive (not recommended)

---

## How to Use the New Parameters

### Command-Line Arguments

You can now fine-tune Stage 2 behavior with new CLI arguments:

```bash
# Example with custom parameters
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    function/train_psnet_two_stage_dist.py \
    --cfg configs/your_config.yaml \
    --stage2_lr 0.00001 \           # Custom LR (default: base_lr * 0.05)
    --stage2_warmup_epochs 8 \       # Custom warmup length (default: auto)
    --sampling_temperature 2.5 \     # Custom sampling (default: 2.0)
    --stage2_epochs 30               # More epochs with conservative LR
```

### Recommended Starting Configuration

For your next run, I recommend:

```bash
# Conservative, stable configuration
--stage2_epochs 30                # Longer training with lower LR
--stage2_lr 0.00002              # Very conservative (or use default 0.05x)
--stage2_warmup_epochs 10        # Full warmup period
--sampling_temperature 2.0       # Balanced sampling (default)
```

### Parameter Tuning Strategy

If you still see instability:
1. **Increase `sampling_temperature`** to 2.5 or 3.0 (more conservative sampling)
2. **Reduce `stage2_lr`** by another 50% (try 0.00001)
3. **Extend `stage2_warmup_epochs`** to 15

If training is too slow:
1. **Slightly increase `stage2_lr`** (try 0.00004)
2. **Reduce `stage2_warmup_epochs`** to 7-8
3. Keep `sampling_temperature` at 2.0 for safety

---

## Expected Training Behavior

### With These Fixes, You Should See:

**Epoch 1-10 (Warmup):**
- Gradual LR increase from 0.000025 → 0.000025 (10% → 100% of base stage2_lr)
- Slow, steady mIoU changes (±0.5-1.0% per epoch)
- No catastrophic drops
- Loss should decrease smoothly

**Epoch 11-30 (Full Training):**
- Stable performance at full (but conservative) LR
- Gradual improvement in tail class performance
- Overall mIoU should match or exceed Stage 1 baseline
- Class-balanced sampling shows moderate, not extreme, oversampling

**Final Results:**
- Stage 2 best mIoU ≥ Stage 1 best mIoU
- Significant improvement on tail classes (reducer, strainer, pump)
- Maintained or improved performance on head classes

---

## Monitoring Your Training

### Key Metrics to Watch

1. **Learning Rate Progression:**
   ```
   Epoch 1/30: Warmup LR = 0.000002 (factor=0.100)
   Epoch 5/30: Warmup LR = 0.000012 (factor=0.500)
   Epoch 10/30: Warmup LR = 0.000025 (factor=1.000)
   Epoch 11/30: Full LR = 0.000025 (warmup complete)
   ```

2. **mIoU Stability:**
   - Should NOT drop more than 2-3% from Stage 1 baseline during warmup
   - Should start improving after epoch 10-15
   - Final mIoU should exceed Stage 1 baseline

3. **Class-Balanced Sampling Statistics:**
   ```
   Stage 2 Epoch 1 - Class-Balanced Sampling Statistics:
   strainer  : ~250 samples (12-15%)  ← Much better than 45%!
   reducer   : ~200 samples (10-12%)  ← Much better than 35%!
   duct      :  ~50 samples (2-3%)    ← Better than 0.2%
   ```

### Warning Signs

**Stop training if you see:**
- mIoU drops >5% in a single epoch
- Training loss increases for 3+ consecutive epochs
- GPU memory errors (reduce batch_size if needed)

---

## Technical Justification

### Why These Values?

1. **0.05× LR multiplier:**
   - Based on transfer learning best practices
   - Fine-tuning typically uses 10-100× smaller LR than pretraining
   - Our 0.025× effective LR (0.05 × 0.5) is in the sweet spot

2. **Temperature = 2.0:**
   - Mathematically: `weight_new = weight_old^(1/2) = sqrt(weight_old)`
   - Example: If inverse frequency gives weight=100, temperature=2.0 → weight=10
   - Reduces extreme oversampling while still helping tail classes

3. **10-epoch warmup (50% of training):**
   - Aligns with fine-tuning literature recommendations
   - Gives classifier time to adapt without feature drift
   - Critical when encoder is frozen and data distribution shifts

---

## Files Modified

```
✓ LTL/two_stage_trainer.py
  - Lines 75-80: LR and warmup configuration
  - Lines 340-346: Dynamic warmup calculation
  - Lines 299-315: Temperature parameter passing
  - Lines 501-519: Conservative warmup schedule
  - Lines 624-629: Updated optimizer creation with logging

✓ LTL/cb_sampler.py
  - Lines 28-63: Added temperature parameter to __init__
  - Lines 127-154: Temperature scaling in weight calculation
  - Lines 350-423: Updated create_balanced_dataloader

✓ function/train_psnet_two_stage_dist.py
  - Lines 53-54: New CLI arguments
  - Lines 96-101: Config parameter setup
```

---

## Next Steps

1. **Resume your training** with the new defaults (just run your existing command)
2. **Monitor the first 5 epochs** - you should see stable, gradual improvement
3. **Check sampling statistics** - verify oversampling is more moderate
4. **If still unstable**, increase `sampling_temperature` to 2.5 or 3.0

## Questions?

These fixes are based on:
- Analysis of your specific training log
- Transfer learning & fine-tuning best practices  
- Long-tailed learning research (decoupled training, BBN, etc.)
- Empirical evidence of what causes catastrophic forgetting

Good luck with your training! 🚀

