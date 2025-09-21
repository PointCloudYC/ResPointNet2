#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
num_gpus=2
dataset_name='psnet12'
log_root="log/${dataset_name}"

echo "================================================="
echo "Starting PSNet12 CB Loss Comparison Experiments"
echo "This will run:"
echo "1. Baseline (standard MaskedCrossEntropy)"
echo "2. CB Loss (beta=0.9999, gamma=2.0, alpha=1.0)"
echo "3. Strong CB Loss (beta=0.99999, gamma=3.0, alpha=2.0)"
echo "================================================="

# --- Experiment 1: Baseline ---
echo ""
echo "=== EXPERIMENT 1: BASELINE ==="
config_name="respointnet2_dp_fi_df_fc1_max"
master_port_train=12351
master_port_eval=12352

echo "Starting baseline training..."
time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml

echo "Baseline training finished. Starting evaluation..."
latest_log_dir=$(ls -dt ${log_root}/${config_name}_* | head -1)
check_point_file="${latest_log_dir}/best.pth"

time python -m torch.distributed.launch --master_port ${master_port_eval} \
    --nproc_per_node 1 \
    function/evaluate_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --load_path "${check_point_file}"

echo "Baseline experiment completed."

# --- Experiment 2: CB Loss ---
echo ""
echo "=== EXPERIMENT 2: CB LOSS ==="
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss"
master_port_train=12353
master_port_eval=12354

echo "Starting CB Loss training..."
time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --use_cb_loss

echo "CB Loss training finished. Starting evaluation..."
latest_log_dir=$(ls -dt ${log_root}/${config_name}_* | head -1)
check_point_file="${latest_log_dir}/best.pth"

time python -m torch.distributed.launch --master_port ${master_port_eval} \
    --nproc_per_node 1 \
    function/evaluate_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --load_path "${check_point_file}" \
    --use_cb_loss

echo "CB Loss experiment completed."

# --- Experiment 3: Strong CB Loss ---
echo ""
echo "=== EXPERIMENT 3: STRONG CB LOSS ==="
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_strong"
master_port_train=12355
master_port_eval=12356

echo "Starting Strong CB Loss training..."
time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --use_cb_loss

echo "Strong CB Loss training finished. Starting evaluation..."
latest_log_dir=$(ls -dt ${log_root}/${config_name}_* | head -1)
check_point_file="${latest_log_dir}/best.pth"

time python -m torch.distributed.launch --master_port ${master_port_eval} \
    --nproc_per_node 1 \
    function/evaluate_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --load_path "${check_point_file}" \
    --use_cb_loss

echo "Strong CB Loss experiment completed."

echo ""
echo "================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "Check the log directories for results:"
echo "- Baseline: ${log_root}/respointnet2_dp_fi_df_fc1_max_*"
echo "- CB Loss: ${log_root}/respointnet2_dp_fi_df_fc1_max_cb_loss_*"
echo "- Strong CB Loss: ${log_root}/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_*"
echo "================================================="
