# Training Visualization Integration Guide

## Overview

Comprehensive visualization and monitoring features have been added to the two-stage trainer. This document explains what was added and how to use it.

## Features Added

### 1. **Automatic Metrics Tracking**
- Training loss per epoch
- Validation loss per epoch  
- mIoU per epoch
- Per-class IoU over time
- Learning rate schedule
- Time per epoch
- Total training time

### 2. **Visualization Plots** (Auto-generated in `log/plots/`)
- `loss_curves.png` - Training and validation loss curves for both stages
- `miou_curve.png` - mIoU progression with best result annotation
- `per_class_iou_heatmap.png` - Heatmap showing IoU evolution for each class
- `learning_rate_schedule.png` - LR schedule across both stages
- `training_time_analysis.png` - Time per epoch and total time breakdown
- `training_summary.png` - Combined summary dashboard with all key metrics

### 3. **Training Report** (`training_report.txt`)
- Text-based summary of training statistics
- Stage-by-stage breakdown
- Per-class IoU results
- Training time analysis
- Checkpoint source tracking

### 4. **Enhanced Logging**
- Checkpoint source logged when using pretrained models
- Training start/end times for each stage
- Parameter count and freeze verification
- Class-balanced sampler statistics

## File Changes

### Modified: `LTL/two_stage_trainer.py`

#### Added Imports:
```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
```

#### Added Class Attributes:
```python
self.plots_dir = os.path.join(config.log_dir, 'plots')
self.stage1_metrics = {
    'train_loss': [], 'val_loss': [], 'miou': [],
    'per_class_iou': [], 'epochs': [], 
    'time_per_epoch': [], 'learning_rates': []
}
self.stage2_metrics = { ... } # Same structure
self.stage1_start_time = None
self.stage1_end_time = None
self.stage2_start_time = None
self.stage2_end_time = None
self.checkpoint_source = None
```

#### Added Methods:
- `_plot_training_curves()` - Main plotting coordinator
- `_plot_loss_curves()` - Loss visualization
- `_plot_miou_curves()` - mIoU progression
- `_plot_per_class_iou_heatmap()` - Per-class IoU heatmap
- `_plot_learning_rate_schedule()` - LR schedule
- `_plot_training_time_analysis()` - Time analysis
- `_plot_combined_summary()` - Combined dashboard
- `_generate_training_report()` - Text report generation

## Integration Points

### Stage 1 Training Loop

Add after each epoch in `train_stage1()`:

```python
# Track metrics
self.stage1_metrics['epochs'].append(epoch)
self.stage1_metrics['train_loss'].append(loss)
self.stage1_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
self.stage1_metrics['time_per_epoch'].append(time.time() - tic)

# After validation
if epoch % self.config.val_freq == 0:
    miou = validate_func(...)
    self.stage1_metrics['miou'].append(miou)
    # Get per-class IoU if available from validation
    # self.stage1_metrics['per_class_iou'].append(per_class_iou)
```

At the end of `train_stage1()`:

```python
self.stage1_end_time = datetime.now()
self.logger.info(f"Stage 1 completed at: {self.stage1_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
self.logger.info(f"Stage 1 duration: {(self.stage1_end_time - self.stage1_start_time).total_seconds() / 3600:.2f} hours")
```

### Stage 2 Training Loop

Add after each epoch in `train_stage2()`:

```python
# Track metrics
self.stage2_metrics['epochs'].append(epoch)
self.stage2_metrics['train_loss'].append(loss)
self.stage2_metrics['learning_rates'].append(stage2_optimizer.param_groups[0]['lr'])
self.stage2_metrics['time_per_epoch'].append(time.time() - tic)

# After validation
if epoch % self.config.val_freq == 0:
    miou = validate_func(...)
    self.stage2_metrics['miou'].append(miou)
    # Get per-class IoU if available
    # self.stage2_metrics['per_class_iou'].append(per_class_iou)
```

At the end of `train_stage2()`:

```python
self.stage2_end_time = datetime.now()
self.logger.info(f"Stage 2 completed at: {self.stage2_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
self.logger.info(f"Stage 2 duration: {(self.stage2_end_time - self.stage2_start_time).total_seconds() / 3600:.2f} hours")

# Generate all plots and reports
self._plot_training_curves()
self._generate_training_report()
```

### Checkpoint Source Logging

When loading a pretrained checkpoint (in Stage 2 start):

```python
stage1_best_path = os.path.join(self.stage1_dir, 'stage1_best.pth')
if os.path.exists(stage1_best_path):
    self._load_checkpoint(stage1_best_path)
    self.checkpoint_source = stage1_best_path
    self.logger.info(f"✓ Loaded Stage 1 checkpoint from: {stage1_best_path}")
```

### Validation Function Integration

To track per-class IoU, the validation function needs to return it. Modify `validate()` to return both mIoU and per-class IoU:

```python
def validate(epoch, test_loader, model, criterion, running_vote_logits, config, num_votes=10):
    # ... existing validation code ...
    
    IoUs, mIoU = seg_metrics(...)
    
    return mIoU, IoUs  # Return both overall and per-class
```

Then in training loops:

```python
miou, per_class_iou = validate_func(...)
self.stage1_metrics['miou'].append(miou)
self.stage1_metrics['per_class_iou'].append(per_class_iou)
```

## Usage Example

The visualization is now automatic! When you run training:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    function/train_psnet_two_stage_dist.py \
    --cfg cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml \
    ...
```

After training completes, check:

```
log/psnet12/experiment_name/
├── plots/
│   ├── loss_curves.png
│   ├── miou_curve.png
│   ├── per_class_iou_heatmap.png
│   ├── learning_rate_schedule.png
│   ├── training_time_analysis.png
│   └── training_summary.png
├── training_report.txt
├── stage1/
│   └── ...
└── stage2/
    └── ...
```

## What Each Plot Shows

### 1. loss_curves.png
- Left panel: Stage 1 train/val loss
- Right panel: Stage 2 train/val loss  
- Shows convergence and overfitting trends

### 2. miou_curve.png
- Combined mIoU progression across both stages
- Green dashed line marks stage transition
- Yellow annotation shows best result

### 3. per_class_iou_heatmap.png
- Rows: Classes
- Columns: Epochs (labeled S1-E1, S1-E2, ..., S2-E1, ...)
- Color: Red (0%) → Yellow (50%) → Green (100%)
- Quickly identify which classes improve/degrade

### 4. learning_rate_schedule.png
- Shows actual LR used at each epoch
- Helps verify scheduler is working correctly
- Stage transition marked

### 5. training_time_analysis.png
- Left: Bar chart of time per epoch
- Right: Pie chart of total time split between stages

### 6. training_summary.png
- Comprehensive dashboard with all key metrics
- Top: Loss and mIoU curves
- Middle: Final per-class IoU bar chart (color-coded)
- Bottom: Statistics, stage comparison

## Troubleshooting

### Plots not generated?
- Check you're on rank 0 (plots only generate on main process)
- Check for matplotlib errors in logs
- Ensure `plots/` directory exists

### Missing metrics?
- Verify you added metric tracking in training loops
- Check that validation function returns per-class IoU

### Wrong checkpoint source?
- Make sure `self.checkpoint_source` is set when loading external checkpoints
- Check the "RESUMING TWO-STAGE TRAINING" section in logs

## Benefits

1. **Easy Monitoring**: Visually track training progress
2. **Quick Debugging**: Identify issues (overfitting, poor classes, etc.)
3. **Documentation**: Auto-generated training reports
4. **Comparison**: Easy to compare experiments
5. **Presentation**: Publication-ready figures

## Next Steps

1. Integrate metric tracking into training loops (see above)
2. Run experiment
3. Check plots in `log/plots/`
4. Share results! 📊
