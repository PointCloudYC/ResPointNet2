# Quick Experiment Guide
## How to Run CB Loss Experiments Immediately

**Last Updated**: 2025-10-01

---

## 🚀 Quick Start (5 Minutes)

### 1. Verify Your Setup

```bash
cd /media/yinchao/code/ResPointNet2

# Check if data is ready
ls data/psnet12/  # Should show train/val splits

# Check if configs exist
ls cfgs/psnet12/*.yaml | grep cb_loss

# Make experiment script executable (if not already)
chmod +x scripts/comprehensive_CB_loss_experiments.sh
```

### 2. Run a Quick Test (10 epochs)

Before running full experiments, test that everything works:

```bash
# Test LD-CB loss with 10 epochs
python -m torch.distributed.launch --master_port 12350 \
    --nproc_per_node 2 \
    function/train_psnet_dist.py \
    --dataset_name psnet12 \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
    --epochs 10 \
    --early_stopping_patience 5
```

**Expected output**:
- Should see "=== Local Density-Aware CB Loss Initialization ===" with class statistics
- Training should start without errors
- Should complete in ~30 minutes

### 3. Run Your First Real Experiment

If the test passes, run the primary contribution:

```bash
# Method 1: Using the comprehensive script
bash scripts/comprehensive_CB_loss_experiments.sh 3

# Method 2: Direct command
python -m torch.distributed.launch --master_port 12350 \
    --nproc_per_node 2 \
    function/train_psnet_dist.py \
    --dataset_name psnet12 \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
    --early_stopping_patience 20
```

**This will take**: ~2-3 days to complete

---

## 📊 Experiment Sequence Recommendation

### Week 1: Core Methods (CRITICAL)

Run these in parallel if you have 2 GPUs:

**GPU 0**: Baseline
```bash
python -m torch.distributed.launch --master_port 12350 --nproc_per_node 1 \
    function/train_psnet_dist.py \
    --dataset_name psnet12 \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml \
    --cb_loss_type none
```

**GPU 1**: LD-CB (Primary Contribution)
```bash
python -m torch.distributed.launch --master_port 12351 --nproc_per_node 1 \
    function/train_psnet_dist.py \
    --dataset_name psnet12 \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml
```

### Week 2: Secondary Method + Key Ablations

**Day 1-2**: GC-CB Loss
```bash
bash scripts/comprehensive_CB_loss_experiments.sh 4
```

**Day 3-7**: LD-CB Radius Ablation
```bash
bash scripts/comprehensive_CB_loss_experiments.sh 5
```

### Week 3: Additional Ablations

**Day 1-5**: LD-CB Weight Ablation
```bash
bash scripts/comprehensive_CB_loss_experiments.sh 6
```

**Day 6-7**: Start writing paper with available results

---

## 🎯 Priority Experiments (If Time Is Limited)

If you can only run **5 experiments** before the deadline:

### Minimum Viable Paper

1. **Baseline (CE Loss)** - Essential comparison ⭐⭐⭐
2. **Standard CB Loss** - Already done! ✅
3. **LD-CB (default params)** - Your main contribution ⭐⭐⭐
4. **LD-CB (radius=0.3)** - Show hyperparameter effect ⭐⭐
5. **LD-CB (radius=0.7)** - Complete ablation ⭐⭐

**This gives you**:
- Core comparison (CE vs CB vs LD-CB)
- One key ablation study
- Total time: ~10 days

### If You Have 10 Experiments

Add these:

6. **GC-CB (default params)** - Secondary contribution ⭐⭐
7. **LD-CB (density_weight=0.5)** - Density modulation effect ⭐⭐
8. **LD-CB (density_weight=1.5)** - Complete ablation ⭐
9. **Seed robustness** (3 seeds for main method) - Statistical significance ⭐⭐
10. **Beta comparison** (CB hyperparameter) - Thorough analysis ⭐

**Total time**: ~20 days

---

## 📝 Logging and Monitoring

### Where to Find Results

All results are saved in:
```
log/psnet12/
  ├── respointnet2_dp_fi_df_fc1_max_20250926015005/     # Baseline
  ├── respointnet2_dp_fi_df_fc1_max_ld_cb_loss_20251001.../  # LD-CB
  └── respointnet2_dp_fi_df_fc1_max_gc_cb_loss_20251001.../  # GC-CB
```

### What to Monitor

1. **Training Log** (`log_train.txt`):
   - Loss curves
   - Learning rate schedule
   - Training time per epoch

2. **Validation Log** (integrated in main log):
   - mIoU per epoch
   - Per-class IoU breakdown
   - Best model performance

3. **Checkpoints**:
   - `best.pth` - Best validation mIoU
   - `current.pth` - Latest checkpoint (for resuming)
   - `ckpt_epoch_X.pth` - Periodic checkpoints

### Quick Check Commands

```bash
# Check latest mIoU
tail -100 log/psnet12/[latest_dir]/log_train.txt | grep "mIoU"

# Monitor training in real-time
watch -n 60 'tail -50 log/psnet12/[latest_dir]/log_train.txt'

# Check if training is still running
nvidia-smi  # Should show GPU utilization

# Extract best mIoU from all experiments
for dir in log/psnet12/*/; do
    echo "$dir:"
    grep "best mIoU" "$dir/log_train.txt" | tail -1
done
```

---

## ⚙️ Hyperparameter Tuning Guide

### LD-CB Loss Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `ld_density_radius` | 0.5 | 0.3-1.0 | Larger = smoother density, less sensitive to local variations |
| `ld_density_weight` | 1.0 | 0.0-2.0 | 0=disabled, 1=full modulation, >1=aggressive |
| `cb_beta` | 0.9999 | 0.99-0.99999 | Higher = more weight on tail classes |
| `cb_gamma` | 2.0 | 0.0-3.0 | Focal loss strength (0=disabled) |

**Recommended tuning order**:
1. Keep all defaults
2. If tail classes underperform: increase `ld_density_weight` to 1.5
3. If head classes suffer: decrease `ld_density_weight` to 0.5
4. Tune `ld_density_radius` based on scanner resolution

### GC-CB Loss Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `gc_context_k` | 16 | 8-64 | Number of neighbors for context |
| `gc_context_weight` | 1.0 | 0.0-2.0 | Strength of contextual hardness |
| `gc_entropy_temp` | 1.0 | 0.5-2.0 | Temperature for entropy smoothing |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. NCCL Timeout Errors

```bash
# Symptom
RuntimeError: NCCL operations have failed or timed out

# Solution
# Increase timeout (edit in train_psnet_dist.py)
# Or reduce batch_size in config
```

#### 2. Out of Memory

```bash
# Symptom
CUDA out of memory

# Solution
# Reduce batch_size in config file:
batch_size: 4  # Instead of 6
```

#### 3. Density Calculation Takes Too Long

```bash
# Symptom
LD-CB initialization hangs at "Calculating local density..."

# Solution
# Reduce max_points_per_class in ld_cb_loss.py
# Or use larger density_radius (faster computation)
```

#### 4. Can't Find Config File

```bash
# Symptom
FileNotFoundError: cfgs/psnet12/xxx.yaml not found

# Solution
# Check you're in project root
cd /media/yinchao/code/ResPointNet2
# Verify config exists
ls cfgs/psnet12/
```

### Performance Issues

#### Training Too Slow?

1. **Check GPU utilization**: `nvidia-smi -l 1`
   - Should be >80% during training
   - If low, check dataloader num_workers

2. **Check dataloader bottleneck**:
   ```python
   # In config, increase num_workers
   num_workers: 8  # From 4 to 8
   ```

3. **Reduce validation frequency**:
   ```yaml
   val_freq: 20  # From 10 to 20 (validate every 20 epochs)
   ```

#### Results Not Improving?

1. **Check learning rate**: Plot loss curve
   - If not decreasing: LR too low
   - If unstable: LR too high

2. **Check class weights**: Look at LD-CB initialization logs
   - Should see higher weights for tail classes
   - If uniform: something wrong with density calculation

3. **Verify data**: Check class distribution
   ```python
   python analyze_psnet12_distribution.py
   ```

---

## 📈 Results Analysis Scripts

### Extract All mIoU Values

```bash
#!/bin/bash
# save as: scripts/extract_results.sh

echo "Experiment,Best_mIoU,Tail_mIoU,Head_mIoU"

for exp_dir in log/psnet12/*/; do
    exp_name=$(basename "$exp_dir")
    best_miou=$(grep "best mIoU" "$exp_dir/log_train.txt" | tail -1 | grep -oP '\d+\.\d+')
    echo "$exp_name,$best_miou,N/A,N/A"
done
```

### Generate Comparison Table

```python
#!/usr/bin/env python3
# save as: scripts/generate_comparison_table.py

import os
import re
from pathlib import Path

log_dir = Path("log/psnet12")
results = {}

for exp_path in log_dir.iterdir():
    if not exp_path.is_dir():
        continue
    
    log_file = exp_path / "log_train.txt"
    if not log_file.exists():
        continue
    
    with open(log_file) as f:
        content = f.read()
        
    # Extract best mIoU
    matches = re.findall(r'best mIoU.*?(\d+\.\d+)', content)
    if matches:
        results[exp_path.name] = float(matches[-1])

# Print sorted results
print(f"{'Experiment':<60} {'mIoU':>8}")
print("-" * 70)
for exp, miou in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{exp:<60} {miou:>7.2f}%")
```

---

## 🎓 Tips for Success

### Do's ✅

1. **Monitor experiments daily** - Catch issues early
2. **Save experiment logs** - Keep organized records
3. **Document unexpected behaviors** - Useful for paper discussion
4. **Run baseline first** - Establish comparison point
5. **Use version control** - Track config changes

### Don'ts ❌

1. **Don't modify configs mid-experiment** - Results won't be comparable
2. **Don't delete logs prematurely** - You might need them later
3. **Don't run too many experiments in parallel** - Can cause OOM
4. **Don't skip validation** - You need it to select best model
5. **Don't forget to set random seeds** - For reproducibility

---

## 📞 Getting Help

### Internal Resources

- `PAPER_WRITING_GUIDE.md` - Comprehensive paper strategy
- `CB_LOSS_IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `CB loss.md` - Theoretical background

### If Stuck

1. Check the logs in `log/psnet12/[experiment]/`
2. Review error messages carefully
3. Try the quick test (10 epochs) first
4. Check GPU memory and utilization
5. Verify data loading works

---

## ✅ Pre-Submission Checklist

Before running final experiments:

- [ ] Verified data is correct and complete
- [ ] Configs are properly set up
- [ ] Ran successful 10-epoch test
- [ ] Have enough disk space (>500GB recommended)
- [ ] Have enough time (2-4 weeks for full suite)
- [ ] Set up monitoring/logging
- [ ] Documented experiment plan

During experiments:

- [ ] Check experiments daily
- [ ] Save intermediate results
- [ ] Take notes on unexpected behaviors
- [ ] Monitor GPU temperature/usage

After experiments:

- [ ] Extract all results to spreadsheet
- [ ] Generate figures and tables
- [ ] Calculate statistical significance
- [ ] Document hyperparameter choices
- [ ] Archive all logs and checkpoints

---

## 🚀 Ready to Start?

**Recommended First Command** (Safe Test):

```bash
# Run LD-CB for 10 epochs as a test
python -m torch.distributed.launch --master_port 12350 \
    --nproc_per_node 2 \
    function/train_psnet_dist.py \
    --dataset_name psnet12 \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
    --epochs 10 \
    --early_stopping_patience 5
```

**If successful, proceed with**:

```bash
# Run the full experiment
bash scripts/comprehensive_CB_loss_experiments.sh 3
```

**Good luck! 🎉**

---

**Document Version**: 1.0  
**Author**: Research Team  
**For Questions**: Check logs first, then review implementation files

