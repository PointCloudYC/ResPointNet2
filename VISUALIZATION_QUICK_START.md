# Training Visualization - Quick Start Guide

## ✅ What's New

Your two-stage trainer now automatically generates:
- 📊 6 visualization plots
- 📝 Detailed training report
- ⏱️ Time tracking
- 📦 Checkpoint source logging
- 🔢 Parameter count verification

## 🚀 How to Use

**Just run training normally!** Everything is automatic.

```bash
bash scripts/test_cb_sampler_fix.sh
# or
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    function/train_psnet_two_stage_dist.py \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml \
    ...
```

## 📁 Output Files

After training, check:
```
log/<dataset>/<experiment>/
├── plots/                          # 📊 All visualizations here!
│   ├── loss_curves.png            # Train/val loss
│   ├── miou_curve.png             # mIoU progression  
│   ├── per_class_iou_heatmap.png  # Per-class IoU over time
│   ├── learning_rate_schedule.png # LR schedule
│   ├── training_time_analysis.png # Time analysis
│   └── training_summary.png       # Combined dashboard
└── training_report.txt            # 📝 Text summary
```

## 📊 Plot Guide

| Plot | What to Look For |
|------|------------------|
| **loss_curves** | ✓ Decreasing loss<br>❌ Big train-val gap (overfitting) |
| **miou_curve** | ✓ Improvement in Stage 2<br>❌ Flat line (not learning) |
| **per_class_iou_heatmap** | 🟢 Green = good classes<br>🔴 Red = problem classes |
| **learning_rate_schedule** | ✓ Smooth schedule<br>✓ Stage 2 LR lower than Stage 1 |
| **training_time_analysis** | ⏱️ See which epochs take longer |
| **training_summary** | 🎯 Everything in one view! |

## 🔍 What's Tracked

| Metric | Stage 1 | Stage 2 |
|--------|---------|---------|
| Training loss | ✓ | ✓ |
| Validation mIoU | ✓ | ✓ |
| Learning rate | ✓ | ✓ |
| Time per epoch | ✓ | ✓ |
| Total time | ✓ | ✓ |
| Checkpoint source | - | ✓ |
| Parameters (trainable/frozen) | - | ✓ |
| CB sampler stats | - | ✓ |

## 📝 Log Output Examples

### Parameter Verification
```
Parameter Freeze Verification
======================================================================
Total parameters:     25,452,444
Trainable parameters: 2,434,476 (9.56%)   # Only decoder!
Frozen parameters:    23,017,968 (90.44%)  # Encoder frozen
```

### Checkpoint Source
```
✓ Loaded Stage 1 best model from: log/.../stage1/stage1_best.pth
```

### CB Sampler Stats
```
Stage 2 Epoch 1 - Class-Balanced Sampling Statistics
======================================================================
   7. reducer  :   334 samples (16.70%)  # 16.7% despite being 0.01% of data!
   8. strainer :   250 samples (12.50%)  # Tail classes sampled frequently
```

### Training Time
```
Stage 2 Duration: 3.57 hours
Total Training Time: 10.78 hours
```

## ⚡ Quick Actions

### View Training Summary
```bash
# Open the combined dashboard
xdg-open log/psnet12/experiment_name/plots/training_summary.png

# Read text report
cat log/psnet12/experiment_name/training_report.txt
```

### Compare Experiments
```bash
# View multiple summaries side-by-side
ls -d log/psnet12/*/plots/training_summary.png
```

### Check if Visualization Worked
```bash
# Should show 6 PNG files
ls log/psnet12/experiment_name/plots/

# Should exist
ls log/psnet12/experiment_name/training_report.txt
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| No plots directory | Check you're on rank 0 (main process) |
| No plots inside | Check logs for matplotlib errors |
| Missing metrics | Check training completed successfully |
| Wrong checkpoint source | Verify `--load_path` or stage1 checkpoint exists |

## 📚 Documentation

- Full details: `yc/VISUALIZATION_FEATURES_SUMMARY.md`
- Integration guide: `yc/VISUALIZATION_INTEGRATION.md`
- CB sampler fixes: `yc/CB_SAMPLER_FIXES.md`
- Problem analysis: `yc/analysis_s1_ce_s2_ce_issues.md`

## 🎯 Next Steps

1. ✓ Run your experiment
2. ✓ Check `plots/` directory
3. ✓ Open `training_summary.png`
4. ✓ Read `training_report.txt`
5. ✓ Compare with other experiments
6. ✓ Iterate and improve!

## 💡 Pro Tips

- **training_summary.png** is perfect for presentations
- **per_class_iou_heatmap.png** shows which classes improve in Stage 2
- **training_report.txt** is great for copying results into papers
- Compare plots across experiments to see what works best
- Check time analysis to optimize training efficiency

---

**Everything is automatic - just train and check the plots!** 🎉
