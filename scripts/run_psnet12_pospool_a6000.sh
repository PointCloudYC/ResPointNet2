#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Use only physical GPU 4 and 5 (0-based). Inside the job they'll appear as logical 0 and 1.
# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=4,5

num_gpus=2
config_name="pospool_xyz_avg_a6000"
dataset_name="psnet12"
log_root="log/${dataset_name}"
master_port_train=12348  # Different port to avoid conflicts
master_port_eval=12347
patience=${1:-50}  # Early stopping patience (default 50)

# --- 1. Training ---
echo "================================================="
echo "Starting PosPool training for ${dataset_name} with config ${config_name}..."
echo "Using patience: ${patience}"
echo "Using GPUs (physical): 4,5  -> (logical inside job): 0,1"
echo "Launcher: torchrun"
echo "================================================="

time torchrun \
  --nproc_per_node ${num_gpus} \
  --master_port ${master_port_train} \
  function/train_psnet_dist.py \
  --dataset_name ${dataset_name} \
  --cfg cfgs/${dataset_name}/${config_name}.yaml \
  --early_stopping_patience ${patience} \
  --cb_loss_type none

echo "Training finished successfully."

# --- 2. Find latest checkpoint ---
echo "================================================="
echo "Searching for the best checkpoint..."
echo "================================================="

latest_log_dir=$(ls -dt ${log_root}/${config_name}_* 2>/dev/null | head -1)

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
echo "Using GPUs (physical): 4,5  -> (logical inside job): 0 (since nproc_per_node=1)"
echo "Launcher: torchrun"
echo "================================================="

time torchrun \
  --nproc_per_node 1 \
  --master_port ${master_port_eval} \
  function/evaluate_psnet_dist.py \
  --dataset_name ${dataset_name} \
  --cfg cfgs/${dataset_name}/${config_name}.yaml \
  --load_path "${check_point_file}"

echo "================================================="
echo "Evaluation finished."
echo "PosPool experiment complete."
echo "================================================="

