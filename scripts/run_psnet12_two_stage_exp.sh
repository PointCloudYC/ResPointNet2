#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
num_gpus=2
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage"
dataset_name='psnet12'
log_root="log/${dataset_name}"
master_port_train=12347
master_port_eval=12348
patience=${1:-20} # Use the first argument for early stopping patience, default to 20

# Two-stage specific parameters
stage1_epochs=80
stage2_epochs=20
stage2_lr=0.0001
use_cb_loss_stage1=false
use_cb_loss_stage2=true
freeze_encoder=true

# --- 1. Two-Stage Training ---
echo "================================================="
echo "Starting Two-Stage training for ${dataset_name} with config ${config_name}..."
echo "Stage 1 Epochs: ${stage1_epochs}, Stage 2 Epochs: ${stage2_epochs}"
echo "Stage 1 CB Loss: ${use_cb_loss_stage1}, Stage 2 CB Loss: ${use_cb_loss_stage2}"
echo "Freeze Encoder: ${freeze_encoder}"
echo "Using patience: ${patience}"
echo "================================================="

time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_two_stage_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --stage1_epochs ${stage1_epochs} \
    --stage2_epochs ${stage2_epochs} \
    --stage2_lr ${stage2_lr} \
    $([ "$use_cb_loss_stage1" = true ] && echo "--use_cb_loss_stage1") \
    $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss_stage2") \
    $([ "$freeze_encoder" = true ] && echo "--freeze_encoder") \
    --early_stopping_patience ${patience}

echo "Two-Stage training finished successfully."

# --- 2. Find latest checkpoint directories ---
echo "================================================="
echo "Searching for the best checkpoints..."
echo "================================================="

# Find the latest directory for this experiment config
latest_log_dir=$(ls -dt ${log_root}/${config_name}_two_stage_* | head -1)

if [ -z "${latest_log_dir}" ]; then
    echo "Error: Could not find any log directory for this two-stage experiment in ${log_root}."
    exit 1
fi

# Check for Stage 1 and Stage 2 checkpoints
stage1_checkpoint="${latest_log_dir}/stage1/stage1_best.pth"
stage2_checkpoint="${latest_log_dir}/stage2/stage2_best.pth"

echo "Found experiment directory: ${latest_log_dir}"

# --- 3. Evaluation of Stage 1 ---
if [ -f "${stage1_checkpoint}" ]; then
    echo "================================================="
    echo "Evaluating Stage 1 checkpoint..."
    echo "================================================="
    
    time python -m torch.distributed.launch --master_port ${master_port_eval} \
        --nproc_per_node 1 \
        function/evaluate_psnet_two_stage_dist.py \
        --dataset_name ${dataset_name} \
        --cfg cfgs/${dataset_name}/${config_name}.yaml \
        --load_path "${stage1_checkpoint}" \
        --stage stage1
    
    echo "Stage 1 evaluation finished."
else
    echo "Warning: Stage 1 checkpoint not found at ${stage1_checkpoint}"
fi

# --- 4. Evaluation of Stage 2 ---
if [ -f "${stage2_checkpoint}" ]; then
    echo "================================================="
    echo "Evaluating Stage 2 (Final) checkpoint..."
    echo "================================================="
    
    time python -m torch.distributed.launch --master_port ${master_port_eval} \
        --nproc_per_node 1 \
        function/evaluate_psnet_two_stage_dist.py \
        --dataset_name ${dataset_name} \
        --cfg cfgs/${dataset_name}/${config_name}.yaml \
        --load_path "${stage2_checkpoint}" \
        --stage stage2 \
        $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss")
    
    echo "Stage 2 evaluation finished."
else
    echo "Warning: Stage 2 checkpoint not found at ${stage2_checkpoint}"
fi

echo "================================================="
echo "Two-Stage experiment complete."
echo "Results saved in: ${latest_log_dir}"
echo "================================================="
