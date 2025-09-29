#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
num_gpus=2
config_names=("respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage" "respointnet2_dp_fi_df_fc1_max_strong_cb_loss_two_stage")
dataset_name='psnet12'
master_port_train=12347

# Experiment directory that was interrupted
experiment_dir="log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_two_stage_20250923124932"

# Two-stage specific parameters
stage1_epochs=80
stage2_epochs=20
stage2_lr=0.0001
use_cb_loss_stage1=false
use_cb_loss_stage2=true
freeze_encoder=true
patience=20


echo "================================================="
echo "Resuming Two-Stage training from interrupted experiment..."
echo "Experiment directory: ${experiment_dir}"
echo "Stage 1 Epochs: ${stage1_epochs}, Stage 2 Epochs: ${stage2_epochs}"
echo "Stage 1  CB Loss: ${use_cb_loss_stage1}, Stage 2 CB Loss: ${use_cb_loss_stage2}"
echo "Freeze Encoder: ${freeze_encoder}"
echo "================================================="

# Check if Stage 1 checkpoint exists
stage1_checkpoint="${experiment_dir}/stage1/stage1_best.pth"
if [ ! -f "${stage1_checkpoint}" ]; then
    echo "ERROR: Stage 1 checkpoint not found at ${stage1_checkpoint}"
    echo "Cannot resume training!"
    exit 1
fi

# Resume training - the trainer will automatically skip Stage 1 if it finds the checkpoint
# use CB loss or strong CB loss
for config_name in ${config_names[@]}; do
    echo "Resuming Two-Stage training for ${config_name}..."
    echo "================================================="
    time python -m torch.distributed.launch --master_port ${master_port_train} \
        --nproc_per_node ${num_gpus} \
        function/train_psnet_two_stage_dist.py \
        --dataset_name ${dataset_name} \
        --cfg cfgs/${dataset_name}/${config_name}.yaml \
        --stage1_epochs ${stage1_epochs} \
        --stage2_epochs ${stage2_epochs} \
        --stage2_lr ${stage2_lr} \
        --log_dir "${experiment_dir}" \
        $([ "$use_cb_loss_stage1" = true ] && echo "--use_cb_loss_stage1") \
        $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss_stage2") \
        $([ "$freeze_encoder" = true ] && echo "--freeze_encoder") \
        --early_stopping_patience ${patience}
done

echo "Two-Stage training resumed and completed!"