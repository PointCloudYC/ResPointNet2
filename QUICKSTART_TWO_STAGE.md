# Two-Stage Training Quick Start Guide

**🚀 Get started in 3 minutes!**

---

## Prerequisites

- ✅ Stage 1 checkpoint available (or will train from scratch)
- ✅ 2+ GPUs with CUDA available
- ✅ Dataset: PSNet12 prepared in `data/` directory

---

## Quick Start (3 Steps)

### Step 1: Set Your Stage 1 Checkpoint

```bash
cd /media/yinchao/code/ResPointNet2

# Edit this line in scripts/run_two_stage_FIXED.sh
nano scripts/run_two_stage_FIXED.sh

# Find line ~30 and set your checkpoint:
STAGE1_CHECKPOINT="log/psnet12/your_stage1_run/stage1/stage1_best.pth"

# OR leave empty to train from scratch:
STAGE1_CHECKPOINT=""
```

### Step 2: Run Training

```bash
# Just run this single command!
bash scripts/run_two_stage_FIXED.sh
```

### Step 3: Check Results

```bash
# View training report
cat log/psnet12/two_stage_FIXED_*/training_report.txt

# Analyze results
python scripts/analyze_two_stage_results.py log/psnet12/two_stage_FIXED_*/

# View plots
ls log/psnet12/two_stage_FIXED_*/plots/
```

---

## That's It! 🎉

The script automatically:
- ✅ Verifies encoder is frozen
- ✅ Verifies class-balanced sampler works
- ✅ Monitors for catastrophic forgetting
- ✅ Generates plots and reports
- ✅ Saves best checkpoints

---

## Expected Output

```
[INFO] Running pre-flight checks...
[SUCCESS] Config file found
[SUCCESS] Using 2 GPUs
[INFO] Starting two-stage training...

🔍 Encoder Freeze Verification
✅ Encoder freeze verification PASSED

🔍 Verifying Class-Balanced Sampler  
✅ Class distribution is WELL BALANCED

Training Stage 2...
Epoch 1: mIoU 52.6% (baseline: 52.5%)
Epoch 5: mIoU 53.1% (+0.6% improvement) ✅
...

[SUCCESS] Training completed successfully!
```

---

## What If Something Goes Wrong?

### Error: "Stage 1 checkpoint not found"

```bash
# Option 1: Train from scratch
STAGE1_CHECKPOINT=""

# Option 2: Set correct path
STAGE1_CHECKPOINT="log/psnet12/correct_path/stage1/stage1_best.pth"
```

### Error: "CUDA out of memory"

```bash
# Reduce batch size in config
nano cfgs/psnet12/respointnet2_two_stage_FIXED.yaml
# Change: batch_size: 6 → 4 or 3
```

### Warning: "Performance dropping"

This is normal in first few epochs. The script will:
- Automatically stop if drop > 10%
- Save emergency checkpoint
- Log detailed diagnostics

---

## Monitoring During Training

```bash
# Follow logs in real-time
tail -f log/psnet12/two_stage_FIXED_*/log.txt

# Check GPU usage
watch -n 1 nvidia-smi

# Filter for important messages
tail -f log/psnet12/two_stage_FIXED_*/log.txt | grep -E "✅|❌|⚠️|mIoU"
```

---

## Advanced Usage

### Custom Hyperparameters

```bash
# Override config parameters
python -m torch.distributed.run \
    --nproc_per_node=2 \
    function/train_psnet_two_stage_dist.py \
    --cfg cfgs/psnet12/respointnet2_two_stage_FIXED.yaml \
    --stage2_lr 0.000005 \
    --stage2_epochs 50 \
    --sampling_temperature 3.0
```

### Resume Training

If training was interrupted:

```bash
# The script automatically detects existing checkpoints
# Just re-run the same command:
bash scripts/run_two_stage_FIXED.sh
```

### Multiple Experiments

```bash
# Run with different settings
for lr in 0.00001 0.000005; do
    python -m torch.distributed.run \
        --nproc_per_node=2 \
        function/train_psnet_two_stage_dist.py \
        --cfg cfgs/psnet12/respointnet2_two_stage_FIXED.yaml \
        --stage2_lr $lr \
        --log_dir log/psnet12/experiment_lr${lr}
done
```

---

## Files to Review After Training

1. **Training Report**: `training_report.txt` - Overall summary
2. **Full Log**: `log.txt` - Detailed training log  
3. **Plots**: `plots/` directory - Visual analysis
4. **Best Model**: `stage2/stage2_best.pth` - For deployment
5. **Config Used**: `config_used.yaml` - Reproducibility

---

## Expected Training Time

| Stage | Duration | Note |
|-------|----------|------|
| Stage 1 (from scratch) | 12-48 hours | One-time only |
| Stage 1 (from checkpoint) | 0 minutes | Instant |
| Stage 2 | 2-4 hours | Quick fine-tuning |
| **Total (w/ checkpoint)** | **2-4 hours** | **Recommended** |

---

## Need Help?

1. **Read the full guide**: `FIXES_APPLIED_TWO_STAGE.md`
2. **Check previous analysis**: `log/.../TWO_STAGE_ANALYSIS_AND_RECOMMENDATIONS.md`
3. **Review verification logs**: Look for ✅ / ❌ / ⚠️ symbols in log
4. **Run diagnostic**: `python scripts/analyze_two_stage_results.py <log_dir>`

---

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "No improvement" | ✅ Expected! Check if degradation < 1% |
| "Degrading fast" | Reduce LR to 0.000005 |
| "OOM error" | Reduce batch_size to 4 or 3 |
| "Sampler not balanced" | Check `sampling_temperature` (try 2.0-3.0) |
| "Encoder leakage" | Bug! Report immediately |

---

**Ready to start? Run:**

```bash
bash scripts/run_two_stage_FIXED.sh
```

**Good luck! 🚀**

