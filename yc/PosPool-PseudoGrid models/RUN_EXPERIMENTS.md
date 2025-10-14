# Running PosPool and PseudoGrid Experiments on PSNet12

## Quick Start (TL;DR)

```bash
# Activate conda environment
conda activate torch

# Test implementation (optional but recommended)
python test_new_models.py

# Run both experiments sequentially
bash scripts/run_psnet12_all_models_a6000.sh 50
```

## Fixed Issues

### Issue 1: Test Script Failures ✅ FIXED

**Problem**: Original test had two failures:
1. PseudoGrid instantiation failed due to distributed check
2. Forward pass failed because masks were float instead of int

**Solution**: 
1. Updated `models/utlis.py` to check `dist.is_initialized()` before calling `dist.get_rank()`
2. Updated `test_new_models.py` to create int32 masks instead of float masks

**Files Modified**:
- `models/utlis.py` (line 186): Added `dist.is_initialized()` check
- `test_new_models.py` (lines 112-113): Changed mask dtype to int32

## Available Scripts

### Option 1: Run Both Experiments Sequentially (Recommended)

```bash
# Run both PosPool and PseudoGrid with patience=50
bash scripts/run_psnet12_all_models_a6000.sh 50

# Run only PosPool
bash scripts/run_psnet12_all_models_a6000.sh 50 1 0

# Run only PseudoGrid
bash scripts/run_psnet12_all_models_a6000.sh 50 0 1

# Custom patience for each
bash scripts/run_psnet12_all_models_a6000.sh 100 1 1
```

**Features**:
- ✅ Colored output for easy reading
- ✅ Automatic checkpoint detection
- ✅ Integrated evaluation after training
- ✅ Comprehensive error handling
- ✅ Execution time tracking
- ✅ Results summary at the end

### Option 2: Run Individual Experiments

```bash
# PosPool only
bash scripts/run_psnet12_pospool_a6000.sh 50

# PseudoGrid only
bash scripts/run_psnet12_pseudogrid_a6000.sh 50
```

## Before Running

### 1. Activate Environment

```bash
conda activate torch
```

### 2. Verify Setup (Recommended)

```bash
python test_new_models.py
```

Expected output:
```
============================================================
PosPool & PseudoGrid Implementation Test Suite
============================================================

... [test outputs] ...

✅ PASS: Config Loading
✅ PASS: GPU Availability
✅ PASS: PosPool Instantiation
✅ PASS: PseudoGrid Instantiation
✅ PASS: Forward Pass

Total: 5/5 tests passed

🎉 All tests passed! Ready to start training.
```

### 3. Check GPU Availability

```bash
nvidia-smi
```

Make sure GPUs 4 and 5 are free.

## Experiment Configurations

### PosPool XYZ (pospool_xyz_avg_a6000.yaml)
- Position embedding: XYZ coordinates
- Reduction: Average pooling
- Optimized for A6000: batch_size=16, amp_opt_level='O1'
- Expected training time: 24-36 hours

### PseudoGrid (pseudo_grid_a6000.yaml)
- Kernel points: 15
- KP influence: Linear
- Convolution mode: Sum
- Optimized for A6000: batch_size=16, amp_opt_level='O1'
- Expected training time: 30-42 hours

## Monitoring Progress

### Option 1: Watch Log Files

```bash
# Watch training progress
tail -f log/psnet12/pospool_xyz_avg_a6000_*/train.log

# Or for PseudoGrid
tail -f log/psnet12/pseudo_grid_a6000_*/train.log
```

### Option 2: TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir log/psnet12/ --port 6006

# Open in browser
# http://localhost:6006
```

### Option 3: Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
```

## Understanding Output

### Training Output

```
Train: [epoch/total][batch/total_batches]
  T: batch time (avg)
  DT: data loading time (avg)
  loss: current loss (avg)
```

### Validation Output

```
E{epoch} V{vote} * mIoU {value}
  - mIoU: mean Intersection over Union (main metric)
  - Higher is better
```

### Early Stopping

```
mIoU did not improve. Current best: XX.XX%. Patience: N/50
```

Training stops after patience epochs without improvement.

## Results Location

After training completes:

```
log/psnet12/
├── pospool_xyz_avg_a6000_TIMESTAMP/
│   ├── best.pth          # Best checkpoint (use this!)
│   ├── current.pth       # Latest checkpoint
│   ├── config.json       # Configuration used
│   ├── ckpt_epoch_*.pth  # Periodic checkpoints
│   └── events.out.tf*    # TensorBoard logs
│
└── pseudo_grid_a6000_TIMESTAMP/
    ├── best.pth
    ├── current.pth
    ├── config.json
    ├── ckpt_epoch_*.pth
    └── events.out.tf*
```

## Collecting Results

### Extract Final mIoU

```bash
# PosPool results
grep "mIoU" log/psnet12/pospool_xyz_avg_a6000_*/train.log | tail -20

# PseudoGrid results
grep "mIoU" log/psnet12/pseudo_grid_a6000_*/train.log | tail -20
```

### Compare with ResPointNet++

```bash
# All model results
for dir in log/psnet12/*/; do
    echo "=== $(basename $dir) ==="
    grep "Best mIoU" $dir/train.log 2>/dev/null || echo "Training in progress..."
done
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
# Edit config file
batch_size: 12  # Instead of 16
```

**Solution 2**: Use fewer GPUs
```bash
export CUDA_VISIBLE_DEVICES=4  # Use only GPU 4
# Then edit script to use --nproc_per_node 1
```

### Issue: Port Already in Use

```bash
# Check what's using the port
lsof -i :12348

# Kill the process or use different ports in script
```

### Issue: Slow Data Loading

```bash
# Check if data is on fast storage
df -h data/

# Reduce num_workers if I/O bottleneck
# Edit config: num_workers: 4
```

### Issue: Training Hangs at Start

This is normal! PseudoGrid creates kernel points on first use:
- Takes ~5 minutes on first run
- Creates files in `kernels/dispositions/`
- Subsequent runs will load these files (instant)

Look for message:
```
============================>[kernel_file] loaded <=======================
```

### Issue: Test Script Fails

Make sure conda environment is activated:
```bash
conda activate torch
python test_new_models.py
```

If tests still fail, check:
1. CUDA operators compiled: `ls ops/pt_custom_ops/*.so`
2. GPU accessible: `nvidia-smi`
3. Config files exist: `ls cfgs/psnet12/*a6000.yaml`

## Timeline Estimate

| Task | Duration | Notes |
|------|----------|-------|
| Setup & Testing | 10 min | One-time |
| PosPool Training | 24-36 h | With early stopping |
| PosPool Evaluation | 30 min | Automatic |
| PseudoGrid Training | 30-42 h | With early stopping |
| PseudoGrid Evaluation | 30 min | Automatic |
| **Total** | **~3-4 days** | Sequential execution |

## Advanced Usage

### Resume Training from Checkpoint

```bash
torchrun --nproc_per_node 2 --master_port 12348 \
  function/train_psnet_dist.py \
  --dataset_name psnet12 \
  --cfg cfgs/psnet12/pospool_xyz_avg_a6000.yaml \
  --load_path log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/current.pth \
  --log_dir log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP
```

### Evaluate Specific Checkpoint

```bash
torchrun --nproc_per_node 1 --master_port 12345 \
  function/evaluate_psnet_dist.py \
  --dataset_name psnet12 \
  --cfg cfgs/psnet12/pospool_xyz_avg_a6000.yaml \
  --load_path log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/ckpt_epoch_100.pth
```

### Run with Different CB Loss

```bash
# With standard CB loss
bash scripts/run_psnet12_pospool_a6000.sh 50

# Then edit the script to change:
# --cb_loss_type none  →  --cb_loss_type standard
```

## Next Steps After Completion

1. **Collect Results**
   ```bash
   # Create summary table
   echo "Model,mIoU,Training Time" > results.csv
   # Add your results
   ```

2. **Update Todos**
   ```bash
   # Mark experiments as complete in yc/todos.md
   ```

3. **Visualize Results**
   ```bash
   # Use TensorBoard for learning curves
   tensorboard --logdir log/psnet12/
   ```

4. **Generate Paper Figures**
   ```bash
   # Export TensorBoard data or create custom plots
   ```

## Summary of All Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_new_models.py` | Verify implementation | `python test_new_models.py` |
| `run_psnet12_pospool_a6000.sh` | Train PosPool | `bash scripts/run_psnet12_pospool_a6000.sh 50` |
| `run_psnet12_pseudogrid_a6000.sh` | Train PseudoGrid | `bash scripts/run_psnet12_pseudogrid_a6000.sh 50` |
| `run_psnet12_all_models_a6000.sh` | Train both models | `bash scripts/run_psnet12_all_models_a6000.sh 50` |

---

**Need Help?**
- Technical details: Check `models/local_aggregation_operators.py`
- Config options: See `cfgs/psnet12/*.yaml`
- Training code: Review `function/train_psnet_dist.py`

