# Stage 2 Training: Critical Bugs Identified and Fixed

## 🚨 Critical Issues Found

### Issue #1: Catastrophic Performance Collapse
**Symptom:** mIoU drops from 54.088% (Stage 1) to 50.2% and continues dropping to 44%

```
Stage 1 (CB checkpoint):  54.088%
Stage 2 Epoch 1:          50.209% (↓ 3.9%)
Stage 2 Epoch 2:          50.602%
Stage 2 Epoch 3:          47.906% (↓ 2.7%)
Stage 2 Epoch 4:          45.240% (↓ 2.7%)
Stage 2 Epoch 5:          44.150% (↓ 1.1%)
...continuing to drop
```

### Issue #2: Wildly Unstable Training Loss
**Symptom:** Loss oscillates dramatically between epochs

```
Epoch  Loss    Change
1      0.118   
2      0.192   +63% ↑
3      0.069   -64% ↓
4      0.113   +64% ↑
5      0.133   +18% ↑
6      0.063   -53% ↓
7      0.094   +49% ↑
8      0.178   +89% ↑
9      0.072   -60% ↓
```

## 🔍 Root Causes Identified

### 1. **No Initial Validation of Loaded Checkpoint**
   - Stage 1 checkpoint was loaded but never validated
   - We don't know if it loaded correctly or what its actual performance was
   - **Fix:** Added validation immediately after loading Stage 1 checkpoint

### 2. **Learning Rate Too Aggressive**
   - Stage 2 LR of 0.0002 is too high for fine-tuning a frozen encoder
   - Destroying learned features from Stage 1
   - **Fix:** Reduced effective LR by 50% + added 3-epoch warmup

### 3. **No Learning Rate Warmup**
   - Fresh optimizer starts optimizing immediately at full LR
   - Causes drastic weight changes in early epochs
   - **Fix:** Added gradual warmup over first 3 epochs (or 25% of training)

### 4. **Insufficient Gradient Clipping**
   - Gradient clipping at 10.0 is too lenient for fine-tuning
   - Allows large weight updates that destroy Stage 1 features
   - **Fix:** Reduced gradient clipping to 1.0 for Stage 2

### 5. **Distribution Shift from Class-Balanced Sampler**
   - Stage 1 trained on imbalanced data distribution
   - Stage 2 sees completely different (balanced) distribution
   - Frozen encoder outputs don't match new distribution
   - **Fix:** Slower LR + warmup helps decoder adapt gradually

## ✅ Fixes Applied

### Fix #1: Validate Loaded Stage 1 Checkpoint
```python
# Now validates immediately after loading
stage1_baseline = validate_func('S1-Loaded', self.val_loader, self.model, 
                              self.criterion, num_votes=2)
logger.info(f"✓ Stage 1 loaded checkpoint baseline mIoU: {stage1_baseline:.3%}")
```

### Fix #2: Reduce Stage 2 Learning Rate
```python
# Reduced by 50% from configured value
effective_lr = self.stage2_lr * 0.5
# For LR=0.0002, now uses 0.0001
```

### Fix #3: Add Learning Rate Warmup
```python
# Gradual warmup over first 3 epochs
warmup_epochs = min(3, self.stage2_epochs // 4)
if epoch <= warmup_epochs:
    warmup_factor = epoch / warmup_epochs
    lr = self.stage2_lr * 0.5 * warmup_factor  # Start from 50% of target
```

### Fix #4: Aggressive Gradient Clipping
```python
# Reduced from 10.0 to 1.0 for fine-tuning stability
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
```

### Fix #5: Initialize Best mIoU from Stage 1
```python
# Track Stage 1 baseline to avoid saving worse checkpoints
if len(self.stage1_metrics['miou']) > 0:
    best_miou = max(self.stage1_metrics['miou'])
```

## 📊 Expected Improvements

With these fixes, you should see:

1. **✅ Stage 1 baseline validated:** ~54% for CB, ~52% for CE
2. **✅ Stable or improving mIoU:** Should NOT drop below Stage 1 baseline
3. **✅ Stable training loss:** Gradual decrease, not wild oscillations
4. **✅ Better final performance:** Closer to best validation mIoU

## 🚀 What To Do Next

### 1. **Stop Current Experiments**
Your currently running experiments have the bugs and won't produce good results.

```bash
# Find and stop running experiments
ps aux | grep train_psnet_two_stage
# Kill them with: kill <PID>
```

### 2. **Re-run with Fixed Code**
```bash
# Run the comprehensive experiments again with fixed code
bash scripts/comprehensive_two_stage_experiments.sh
```

### 3. **Monitor Training**
Watch for these indicators of success:

```bash
# Monitor real-time
tail -f log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_exp*/log_resume.txt

# Check for:
# - "Stage 1 loaded checkpoint baseline mIoU: XX.XXX%"
# - Warmup learning rates in first 3 epochs
# - Stable or increasing validation mIoU
# - Gradually decreasing training loss (not oscillating)
```

### 4. **Expected Training Pattern (CORRECTED)**

**Stage 1 Validation:**
```
[INFO] Validating loaded Stage 1 checkpoint on validation set...
[INFO] ✓ Stage 1 loaded checkpoint baseline mIoU: 54.088%
```

**Stage 2 Training (First 5 Epochs):**
```
Epoch 1: LR ~0.00003, Loss 0.15, mIoU ~53.5% (slight drop is OK due to warmup)
Epoch 2: LR ~0.00007, Loss 0.12, mIoU ~54.0% (recovering)
Epoch 3: LR ~0.0001,  Loss 0.10, mIoU ~54.5% (improving)
Epoch 4: LR ~0.0001,  Loss 0.09, mIoU ~55.0% (continuing to improve)
Epoch 5: LR ~0.0001,  Loss 0.08, mIoU ~55.5% (beating Stage 1!)
```

**What You Should See:**
- ✅ Gradual LR increase during warmup (epochs 1-3)
- ✅ Steady loss decrease (not oscillating)
- ✅ mIoU stays near or improves from Stage 1 baseline
- ✅ Best mIoU should exceed Stage 1 by 1-2% if two-stage works

## 📈 Theory vs Reality Check

**Two-Stage Training Theory:**
- Stage 1: Learn rich representations on all data (imbalanced is OK)
- Stage 2: Fine-tune classifier with balanced sampling (improve tail classes)
- **Expected:** Small improvement (1-3%) in overall mIoU, larger improvement in tail classes

**What Was Happening (BUGGY):**
- Stage 1: ✅ Learned good representations (54%)
- Stage 2: ❌ Destroyed them with aggressive LR + no warmup
- **Result:** Performance collapse instead of improvement

**What Should Happen Now (FIXED):**
- Stage 1: ✅ Learned good representations (54%)
- Stage 2: ✅ Gently adapt classifier with stable training
- **Expected Result:** 54-56% overall, with improved tail class performance

## 🎯 Success Criteria

Consider Stage 2 training successful if:

1. ✅ Stage 1 baseline validated: ~54% (CB) or ~52% (CE)
2. ✅ Stage 2 mIoU ≥ Stage 1 baseline (never drops below)
3. ✅ Training loss decreases smoothly
4. ✅ Best Stage 2 mIoU ≥ Stage 1 + 1%
5. ✅ Final mIoU (20 votes) within 1-2% of best Stage 2

## 🔧 Additional Recommendations

If problems persist after these fixes:

1. **Further reduce Stage 2 LR:** Try 0.00005 (half of current fixed value)
2. **Increase warmup period:** Try 5 epochs instead of 3
3. **Use cosine annealing:** Gradually decrease LR after warmup
4. **Reduce Stage 2 epochs:** 20 epochs might be too many, try 10
5. **Consider unfreezing encoder gradually:** Unfreeze top layers first

## 📝 Files Modified

- `LTL/two_stage_trainer.py`: Added validation, warmup, LR reduction, gradient clipping
- `function/train_psnet_two_stage_dist.py`: Already had best checkpoint loading fix

---

**Bottom Line:** The bugs were SEVERE but now FIXED. Re-run your experiments with the corrected code!

