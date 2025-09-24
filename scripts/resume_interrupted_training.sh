#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
num_gpus=2
dataset_name='psnet12'
master_port_train=12347

# EXACT experiment directory that was interrupted
experiment_dir="log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_two_stage_20250923124932"
config_file="cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage.yaml"

# Two-stage specific parameters (same as original training)
stage1_epochs=80
stage2_epochs=20
stage2_lr=0.0001
use_cb_loss_stage1=false
use_cb_loss_stage2=true
freeze_encoder=true
patience=20

echo "================================================="
echo "RESUMING INTERRUPTED TWO-STAGE TRAINING"
echo "================================================="
echo "Experiment directory: ${experiment_dir}"
echo "Config file: ${config_file}"
echo "Stage 1 Epochs: ${stage1_epochs}, Stage 2 Epochs: ${stage2_epochs}"
echo "Stage 2 Learning Rate: ${stage2_lr}"
echo "Stage 1 CB Loss: ${use_cb_loss_stage1}, Stage 2 CB Loss: ${use_cb_loss_stage2}"
echo "Freeze Encoder: ${freeze_encoder}"
echo "================================================="

# Check if Stage 1 checkpoint exists
stage1_checkpoint="${experiment_dir}/stage1/stage1_best.pth"
if [ ! -f "${stage1_checkpoint}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${stage1_checkpoint}"
    echo "Cannot resume training!"
    exit 1
fi

echo "✓ Found Stage 1 checkpoint: ${stage1_checkpoint}"
echo "Starting Stage 2 training..."
echo "================================================="

# Resume training - will automatically skip Stage 1 and go to Stage 2
time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_two_stage_dist.py \
    --dataset_name ${dataset_name} \
    --cfg ${config_file} \
    --stage1_epochs ${stage1_epochs} \
    --stage2_epochs ${stage2_epochs} \
    --stage2_lr ${stage2_lr} \
    --log_dir "${experiment_dir}" \
    $([ "$use_cb_loss_stage1" = true ] && echo "--use_cb_loss_stage1") \
    $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss_stage2") \
    $([ "$freeze_encoder" = true ] && echo "--freeze_encoder") \
    --early_stopping_patience ${patience}

echo "================================================="
echo "Two-Stage training resumed and completed!"
echo "Check results in: ${experiment_dir}"
echo "================================================="
