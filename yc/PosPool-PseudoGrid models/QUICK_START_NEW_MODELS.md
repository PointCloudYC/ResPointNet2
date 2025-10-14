# Quick Start: PosPool and PseudoGrid Models

This guide provides quick commands to train and evaluate the newly added PosPool and PseudoGrid models on PSNet12 dataset.

## Prerequisites

Ensure you have:
- ✅ Conda environment activated (same as ResPointNet++)
- ✅ CUDA operators compiled
- ✅ PSNet12 data prepared in `data/` directory
- ✅ Access to GPUs 4 and 5

## Quick Commands

### 1. PosPool with XYZ Encoding

Train and evaluate in one command:
```bash
bash scripts/run_psnet12_pospool_a6000.sh
```

With custom early stopping patience:
```bash
bash scripts/run_psnet12_pospool_a6000.sh 50
```

### 2. PseudoGrid (KPConv)

Train and evaluate in one command:
```bash
bash scripts/run_psnet12_pseudogrid_a6000.sh
```

With custom early stopping patience:
```bash
bash scripts/run_psnet12_pseudogrid_a6000.sh 50
```

### 3. PosPool with Sin-Cos Encoding (Manual)

To try the sin-cos variant, you need to manually update the config in the script or run training directly:

```bash
export CUDA_VISIBLE_DEVICES=4,5

torchrun --nproc_per_node 2 --master_port 12352 \
  function/train_psnet_dist.py \
  --dataset_name psnet12 \
  --cfg cfgs/psnet12/pospool_sin_cos_avg_a6000.yaml \
  --early_stopping_patience 50 \
  --cb_loss_type none
```

## Expected Outputs

After successful training, you'll find:
```
log/psnet12/<model_name>_<timestamp>/
├── best.pth              # Best model checkpoint (highest val mIoU)
├── current.pth           # Latest model checkpoint
├── config.json           # Training configuration
├── config.yaml           # Copy of original config
├── ckpt_epoch_*.pth      # Periodic checkpoints (every 10 epochs)
└── events.out.tfevents*  # TensorBoard logs
```

## Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir log/psnet12/ --port 6006
```

Then open browser: `http://localhost:6006`

## Common Issues and Solutions

### Issue: "RuntimeError: CUDA out of memory"
**Solution 1**: Reduce batch size in config
```yaml
batch_size: 12  # Instead of 16
```

**Solution 2**: Use single GPU
```bash
export CUDA_VISIBLE_DEVICES=4  # Use only GPU 4
torchrun --nproc_per_node 1 ...  # Change to 1 GPU
```

### Issue: "Port already in use"
**Solution**: Use a different port
```bash
# In the script, change master_port_train to an unused port
master_port_train=12400  # Or any free port
```

### Issue: "FileNotFoundError: kernels/dispositions/..."
**Solution**: This is normal on first run. The script will create kernel points automatically. Wait for the message:
```
=============================>...loaded <======================
```

## Model Comparison Quick Reference

| Model | Training Speed | Memory Usage | Expected mIoU | Config File |
|-------|----------------|--------------|---------------|-------------|
| ResPointNet++ | Fast (baseline) | Moderate | ~XX% | respointnet2_dp_fi_df_fc1_max_a6000.yaml |
| PosPool (xyz) | Moderate (-10%) | Moderate | ~XX% | pospool_xyz_avg_a6000.yaml |
| PosPool (sin_cos) | Moderate (-10%) | Moderate | ~XX% | pospool_sin_cos_avg_a6000.yaml |
| PseudoGrid | Slower (-25%) | Higher | ~XX% | pseudo_grid_a6000.yaml |

*Note: Fill in mIoU values after running experiments*

## Training Time Estimates (2x A6000)

With early stopping (patience=50):
- **PosPool**: 24-36 hours (or less if early stopping triggers)
- **PseudoGrid**: 30-42 hours (or less if early stopping triggers)

Without early stopping (600 epochs):
- **PosPool**: ~48 hours
- **PseudoGrid**: ~60 hours

## Evaluation Only

To evaluate an existing checkpoint:
```bash
export CUDA_VISIBLE_DEVICES=4

torchrun --nproc_per_node 1 --master_port 12345 \
  function/evaluate_psnet_dist.py \
  --dataset_name psnet12 \
  --cfg cfgs/psnet12/pospool_xyz_avg_a6000.yaml \
  --load_path log/psnet12/pospool_xyz_avg_a6000_TIMESTAMP/best.pth
```

## Next Steps

1. **Train all models**:
   ```bash
   # Run in sequence or parallel in different terminals
   bash scripts/run_psnet12_pospool_a6000.sh 50
   bash scripts/run_psnet12_pseudogrid_a6000.sh 50
   ```

2. **Collect results**: Check the final validation metrics in log files

3. **Compare performance**: Use TensorBoard to compare learning curves

4. **Update todos**: Mark models as "evaluated" in `yc/todos.md`

## Getting Help

For detailed information:
- See `docs/NEW_MODELS.md` for comprehensive documentation
- Check `cfgs/psnet12/*.yaml` for all configuration options
- Review `models/local_aggregation_operators.py` for implementation details

For issues specific to the existing codebase:
- Check `yc/todos.md` section "problems" for known issues and solutions

