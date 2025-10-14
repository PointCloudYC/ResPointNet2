#!/bin/bash

# ============================================================================
# Comprehensive Experiment Runner for PSNet12 Dataset on A6000 GPUs
# ============================================================================
# This script runs PosPool and PseudoGrid experiments sequentially
# Usage: bash scripts/run_psnet12_all_models_a6000.sh [patience] [run_pospool] [run_pseudogrid]
# Example: bash scripts/run_psnet12_all_models_a6000.sh 50 1 1
# ============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
export CUDA_VISIBLE_DEVICES=4,5
num_gpus=2
dataset_name="psnet12"
log_root="log/${dataset_name}"

# Parse arguments
patience=${1:-50}  # Early stopping patience (default 50)
run_pospool=${2:-1}  # Whether to run PosPool (default 1=yes)
run_pseudogrid=${3:-1}  # Whether to run PseudoGrid (default 1=yes)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}$1${NC}"
    echo "============================================================"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run experiment
run_experiment() {
    local config_name=$1
    local model_name=$2
    local master_port_train=$3
    local master_port_eval=$4
    
    print_header "Starting ${model_name} Experiment"
    
    print_info "Configuration:"
    print_info "  - Model: ${model_name}"
    print_info "  - Config: ${config_name}.yaml"
    print_info "  - Dataset: ${dataset_name}"
    print_info "  - GPUs: 4,5 (logical 0,1)"
    print_info "  - Patience: ${patience}"
    print_info "  - Master ports: ${master_port_train} (train), ${master_port_eval} (eval)"
    
    # Training
    print_info "Starting training..."
    
    if time torchrun \
      --nproc_per_node ${num_gpus} \
      --master_port ${master_port_train} \
      function/train_psnet_dist.py \
      --dataset_name ${dataset_name} \
      --cfg cfgs/${dataset_name}/${config_name}.yaml \
      --early_stopping_patience ${patience} \
      --cb_loss_type none; then
        print_info "Training completed successfully"
    else
        print_error "Training failed for ${model_name}!"
        return 1
    fi
    
    # Find checkpoint
    print_info "Searching for best checkpoint..."
    latest_log_dir=$(ls -dt ${log_root}/${config_name}_* 2>/dev/null | head -1)
    
    if [ -z "${latest_log_dir}" ]; then
        print_error "Could not find log directory for ${model_name}"
        return 1
    fi
    
    check_point_file="${latest_log_dir}/best.pth"
    
    if [ ! -f "${check_point_file}" ]; then
        print_error "Checkpoint file 'best.pth' not found in ${latest_log_dir}"
        return 1
    fi
    
    print_info "Found checkpoint: ${check_point_file}"
    
    # Evaluation
    print_info "Starting evaluation..."
    
    if time torchrun \
      --nproc_per_node 1 \
      --master_port ${master_port_eval} \
      function/evaluate_psnet_dist.py \
      --dataset_name ${dataset_name} \
      --cfg cfgs/${dataset_name}/${config_name}.yaml \
      --load_path "${check_point_file}"; then
        print_info "Evaluation completed successfully"
    else
        print_error "Evaluation failed for ${model_name}!"
        return 1
    fi
    
    print_header "${model_name} Experiment Complete"
    return 0
}

# Main execution
print_header "PSNet12 Comprehensive Experiment Suite"
print_info "Experiment configuration:"
print_info "  - Early stopping patience: ${patience}"
print_info "  - Run PosPool: ${run_pospool}"
print_info "  - Run PseudoGrid: ${run_pseudogrid}"
print_info "  - GPUs: 4,5"
echo ""

# Track results
declare -a results
total_start_time=$(date +%s)

# Run PosPool experiment
if [ ${run_pospool} -eq 1 ]; then
    if run_experiment "pospool_xyz_avg_a6000" "PosPool (XYZ)" 12348 12347; then
        results+=("PosPool: SUCCESS")
    else
        results+=("PosPool: FAILED")
        print_error "PosPool experiment failed, but continuing..."
    fi
else
    print_info "Skipping PosPool experiment"
    results+=("PosPool: SKIPPED")
fi

# Run PseudoGrid experiment
if [ ${run_pseudogrid} -eq 1 ]; then
    if run_experiment "pseudo_grid_a6000" "PseudoGrid (KPConv)" 12350 12349; then
        results+=("PseudoGrid: SUCCESS")
    else
        results+=("PseudoGrid: FAILED")
        print_error "PseudoGrid experiment failed"
    fi
else
    print_info "Skipping PseudoGrid experiment"
    results+=("PseudoGrid: SKIPPED")
fi

# Calculate total time
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))
seconds=$((total_duration % 60))

# Final summary
print_header "Experiment Suite Complete"
echo ""
echo "Results Summary:"
echo "----------------------------------------"
for result in "${results[@]}"; do
    if [[ $result == *"SUCCESS"* ]]; then
        echo -e "${GREEN}✓${NC} $result"
    elif [[ $result == *"FAILED"* ]]; then
        echo -e "${RED}✗${NC} $result"
    else
        echo -e "${YELLOW}○${NC} $result"
    fi
done
echo "----------------------------------------"
echo ""
printf "Total execution time: %02d:%02d:%02d\n" $hours $minutes $seconds
echo ""
print_info "All logs saved in: ${log_root}/"
print_info "Use TensorBoard to visualize: tensorboard --logdir ${log_root}/ --port 6006"
echo ""
echo "============================================================"

