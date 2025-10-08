# 🚀 Two-Stage Training: Quick Reference Card

## ✅ All Fixed - Ready to Restart!

### **Stop Current Training**
```bash
ps aux | grep train_psnet_two_stage
kill <PID>
```

### **Restart Training**
```bash
cd /media/yinchao/code/ResPointNet2
bash scripts/comprehensive_two_stage_experiments.sh
```

### **Monitor Progress**
```bash
# Watch logs in real-time
tail -f log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_exp*/log_resume.txt

# Quick check latest mIoU
grep "ES2-E.*V1 \* mIoU" log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_exp*/log_resume.txt | tail -10
```

## 🎯 What To Look For (Success Indicators)

### ✅ **Good Signs:**
```
✓ ES1-Loaded V1 * mIoU 50.XXX%         # NOT 0.052%!
✓ Epoch 1/20: Warmup LR = 0.000030     # Gradual warmup
✓ New best mIoU: 51.234% (epoch 3)     # Improving
✓ Loss: 0.145 → 0.125 → 0.110 → ...   # Decreasing
```

### ⚠️ **Warning Signs:**
```
⚠️  Performance dropping! Stage 1: 50.4%, Current: 45.1%
⚠️  Drop of 5.3% detected
```

### ❌ **Critical Failure:**
```
❌ CRITICAL: mIoU dropped >10% below Stage 1 baseline!
❌ Stopping early to prevent further degradation
```

## 📊 Expected Results

| Stage | Checkpoint | Expected mIoU | Status |
|-------|-----------|---------------|--------|
| Stage 1 CE | Pretrained | ~52% | ✅ |
| Stage 1 CB | Pretrained | ~54% | ✅ |
| Stage 2 Epoch 1 | Warmup | 49-51% | ⚠️ OK |
| Stage 2 Epoch 3 | Post-warmup | 51-53% | ✅ Good |
| Stage 2 Epoch 10+ | Converged | 52-56% | ✅ Target |
| Final (20 votes) | Best S2 | Within 1-2% | ✅ |

## 🔧 Fixes Applied

1. ✅ **Removed duplicate validation** (was validating unloaded model = 0.05%)
2. ✅ **Fixed warmup LR** (clearer, more stable)
3. ✅ **Added early stopping** (stops if mIoU drops >10%)
4. ✅ **Better logging** (shows progress clearly)

## 📞 Quick Debug Commands

```bash
# See all experiment directories
ls -lth log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_exp*/

# Check if training is running
ps aux | grep train_psnet | grep -v grep

# See latest validation results
grep "mIoU" log/psnet12/*/log_resume.txt | tail -20

# Check GPU usage
nvidia-smi
```

## ⏱️ Estimated Timeline

- **Per experiment:** ~1.5-2 hours (20 epochs + final validation)
- **Full suite:** ~15-18 hours (9 experiments)
- **Run overnight:** Perfect timing!

---

**Status:** ✅ **READY - All bugs fixed!**
**Action:** Stop current training, restart with fixed code

