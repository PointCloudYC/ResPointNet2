# 🚀 Quick Commands Reference

## TL;DR - Just Run This

```bash
conda activate torch
python test_new_models.py
bash scripts/run_psnet12_all_models_a6000.sh 50
```

## All Commands

### 1. Setup & Testing

```bash
# Activate environment
conda activate torch

# Test implementation (RECOMMENDED)
python test_new_models.py

# Check GPUs
nvidia-smi
```

### 2. Run Experiments

#### Option A: Both Models (Recommended)
```bash
# Run both PosPool and PseudoGrid sequentially
bash scripts/run_psnet12_all_models_a6000.sh 50

# Custom patience
bash scripts/run_psnet12_all_models_a6000.sh 100

# Only PosPool
bash scripts/run_psnet12_all_models_a6000.sh 50 1 0

# Only PseudoGrid
bash scripts/run_psnet12_all_models_a6000.sh 50 0 1
```

#### Option B: Individual Models
```bash
# PosPool only
bash scripts/run_psnet12_pospool_a6000.sh 50

# PseudoGrid only
bash scripts/run_psnet12_pseudogrid_a6000.sh 50
```

### 3. Monitoring

```bash
# TensorBoard
tensorboard --logdir log/psnet12/ --port 6006
# Then open: http://localhost:6006

# Watch training log (PosPool)
tail -f log/psnet12/pospool_xyz_avg_a6000_*/train.log

# Watch training log (PseudoGrid)
tail -f log/psnet12/pseudo_grid_a6000_*/train.log

# GPU usage
watch -n 1 nvidia-smi

# Disk usage
df -h
```

### 4. Results Collection

```bash
# Show all mIoU results
grep "mIoU" log/psnet12/*/train.log | tail -20

# Find best checkpoints
find log/psnet12/ -name "best.pth"

# Compare models
for dir in log/psnet12/*/; do
    echo "=== $(basename $dir) ==="
    grep "Best" $dir/train.log 2>/dev/null || echo "In progress..."
done
```

### 5. Troubleshooting

```bash
# Re-run test
python test_new_models.py

# Check CUDA operators
ls -lah ops/pt_custom_ops/*.so

# Check config files
ls -lah cfgs/psnet12/*a6000.yaml

# Check scripts
ls -lah scripts/*pospool* scripts/*pseudo* scripts/*all*

# Kill stuck processes
pkill -f train_psnet_dist

# Check ports
lsof -i :12348
lsof -i :12350
```

### 6. Advanced Usage

```bash
# Resume from checkpoint
torchrun --nproc_per_node 2 --master_port 12348 \
  function/train_psnet_dist.py \
  --dataset_name psnet12 \
  --cfg cfgs/psnet12/pospool_xyz_avg_a6000.yaml \
  --load_path log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/current.pth \
  --log_dir log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP

# Evaluate specific checkpoint
torchrun --nproc_per_node 1 --master_port 12345 \
  function/evaluate_psnet_dist.py \
  --dataset_name psnet12 \
  --cfg cfgs/psnet12/pospool_xyz_avg_a6000.yaml \
  --load_path log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/best.pth

# Use different CB loss
# Edit script and change: --cb_loss_type none → --cb_loss_type standard
```

## Common Workflows

### Workflow 1: Quick Test & Run
```bash
conda activate torch
python test_new_models.py
bash scripts/run_psnet12_all_models_a6000.sh 50
```

### Workflow 2: Test, Run, Monitor
```bash
# Terminal 1
conda activate torch
python test_new_models.py
bash scripts/run_psnet12_all_models_a6000.sh 50

# Terminal 2
tensorboard --logdir log/psnet12/ --port 6006

# Terminal 3
watch -n 1 nvidia-smi
```

### Workflow 3: Sequential with Delays
```bash
# Run PosPool
bash scripts/run_psnet12_pospool_a6000.sh 50

# Wait for completion, then run PseudoGrid
bash scripts/run_psnet12_pseudogrid_a6000.sh 50
```

## Environment Variables

```bash
# Change GPUs
export CUDA_VISIBLE_DEVICES=4,5    # Default
export CUDA_VISIBLE_DEVICES=0,1    # Use GPUs 0,1
export CUDA_VISIBLE_DEVICES=4      # Single GPU

# Change master ports (if conflicts)
# Edit in scripts:
master_port_train=12348  # Change to any free port
master_port_eval=12347   # Change to any free port
```

## File Locations

```bash
# Configs
cfgs/psnet12/pospool_xyz_avg_a6000.yaml
cfgs/psnet12/pospool_sin_cos_avg_a6000.yaml
cfgs/psnet12/pseudo_grid_a6000.yaml

# Scripts
scripts/run_psnet12_all_models_a6000.sh
scripts/run_psnet12_pospool_a6000.sh
scripts/run_psnet12_pseudogrid_a6000.sh

# Test
test_new_models.py

# Logs (after running)
log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/
log/psnet12/pseudo_grid_a6000_TIMESTAMP/

# Checkpoints (after training)
log/psnet12/*/best.pth
```

## Quick Checks

```bash
# Is conda env active?
which python
# Should show: .../envs/torch/bin/python

# Are CUDA ops compiled?
python -c "import torch; from pt_utils import MaskedQueryAndGroup; print('OK')"

# Can load configs?
python -c "from utils.config import config, update_config; update_config('cfgs/psnet12/pospool_xyz_avg_a6000.yaml'); print('OK')"

# Are GPUs free?
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

## Error Recovery

```bash
# If training crashes
# 1. Check last log
tail -100 log/psnet12/MODEL_NAME_TIMESTAMP/train.log

# 2. Check if checkpoint exists
ls -lah log/psnet12/MODEL_NAME_TIMESTAMP/current.pth

# 3. Resume if checkpoint exists
# Use the resume command from Advanced Usage section

# If port conflict
# 1. Find what's using port
lsof -i :PORT_NUMBER

# 2. Kill process
kill -9 PID

# 3. Or change port in script

# If CUDA OOM
# 1. Reduce batch size in config
# Edit YAML: batch_size: 12  (instead of 16)

# 2. Or use single GPU
export CUDA_VISIBLE_DEVICES=4
# Edit script: --nproc_per_node 1
```

## Results Summary Commands

```bash
# After all experiments complete:

# 1. List all results
ls -lah log/psnet12/*/best.pth

# 2. Extract final mIoU
grep "mIoU" log/psnet12/*/train.log | grep "Best\|Final" | sort

# 3. Create results table
echo "Model,mIoU,Time" > results.csv
# Add your results manually

# 4. Compare with baseline
echo "=== ResPointNet++ ==="
grep "mIoU" log/psnet12/respointnet2_*/train.log | tail -5
echo "=== PosPool ==="
grep "mIoU" log/psnet12/pospool_*/train.log | tail -5
echo "=== PseudoGrid ==="
grep "mIoU" log/psnet12/pseudo_*/train.log | tail -5
```

---

## Most Important Commands

### Before Running
```bash
conda activate torch
python test_new_models.py
```

### Run Everything
```bash
bash scripts/run_psnet12_all_models_a6000.sh 50
```

### Monitor Progress
```bash
tensorboard --logdir log/psnet12/ --port 6006
```

### Get Results
```bash
grep "mIoU" log/psnet12/*/train.log | tail -20
```

---

**See Also:**
- `FINAL_SUMMARY.md` - Quick start guide
- `RUN_EXPERIMENTS.md` - Complete documentation
- `FIXES_APPLIED.md` - Technical details
- `FILES_OVERVIEW.md` - File structure

