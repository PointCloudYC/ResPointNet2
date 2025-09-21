#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
num_gpus=2
config_name="respointnet2_dp_fi_df_fc1_max"
dataset_name='psnet5'
log_root="log/${dataset_name}"
master_port_train=12346
master_port_eval=12345

# --- 1. Training ---
echo "================================================="
echo "Starting training for ${dataset_name} with config ${config_name}..."
echo "================================================="
time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml

echo "Training finished successfully."

# --- 2. Find latest checkpoint ---
echo "================================================="
echo "Searching for the best checkpoint..."
echo "================================================="
# Find the latest directory for this experiment config
latest_log_dir=$(ls -dt ${log_root}/${config_name}_* | head -1)

if [ -z "${latest_log_dir}" ]; then
    echo "Error: Could not find any log directory for this experiment in ${log_root}."
    exit 1
fi

check_point_file="${latest_log_dir}/best.pth"

if [ ! -f "${check_point_file}" ]; then
    echo "Error: Checkpoint file 'best.pth' not found in the latest log directory: ${latest_log_dir}!"
    exit 1
fi

echo "Found checkpoint: ${check_point_file}"

# --- 3. Evaluation ---
echo "================================================="
echo "Starting evaluation..."
echo "================================================="
time python -m torch.distributed.launch --master_port ${master_port_eval} \
    --nproc_per_node 1 \
    function/evaluate_psnet_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --load_path "${check_point_file}"

echo "================================================="
echo "Evaluation finished."
echo "Experiment complete."
echo "================================================="
