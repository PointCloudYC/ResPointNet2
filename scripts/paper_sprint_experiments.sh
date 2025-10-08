#!/bin/bash

################################################################################
# Paper Sprint Core Experiments for PSNet12
#
# This script runs the core experiments required for the paper submission sprint,
# as outlined in yc/ActionPlan-CB.md (Days 1-4).
#
# CHECKPOINT REUSE STRATEGY:
#   Each experiment continues from its own previous checkpoint:
#   - LD-CB checkpoint: 58 epochs trained, 42.77% mIoU, patience 20→50
#     Path: log/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss_20251005034405/best.pth
#   - GC-CB checkpoint: 80 epochs trained, 44.31% mIoU, patience 20→50  ✓ BETTER
#     Path: log/psnet12/respointnet2_dp_fi_df_fc1_max_gc_cb_loss_20251003071346/best.pth
#
# EXPERIMENTS:
#   1. LD-CB Loss (continue LD-CB training, patience=50)
#   2. GC-CB Loss (continue GC-CB training, patience=50)
#   3. Combined Loss (start from better GC-CB checkpoint)
#   4. Ablation: LD-CB radius (r=0.3, 0.7, 1.0; from LD-CB checkpoint)
#   5. Ablation: GC-CB k (k=8, 32, 64; from GC-CB checkpoint)
#
# Usage:
#   bash scripts/paper_sprint_experiments.sh              # Interactive menu
#   bash scripts/paper_sprint_experiments.sh all          # Run all experiments
#   bash scripts/paper_sprint_experiments.sh 1 2 3        # Run specific experiments
#
################################################################################

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

NUM_GPUS=2
DATASET_NAME='psnet12'
LOG_ROOT="log/${DATASET_NAME}"
BASE_PORT=12370  # Starting port, will increment for each experiment
EARLY_STOPPING_PATIENCE=50
declare -i EXPERIMENT_PORT_OFFSET=0 # Use a global counter for port offset

# !!! IMPORTANT: Paths to checkpoints from previous experiments !!!
# Each experiment continues from its own checkpoint (patience increased from 20 to 50)
LDCB_CHECKPOINT="log/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss_20251005034405/best.pth"  # 58 epochs, 42.77% mIoU
LDCB_LOG_DIR="log/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss_20251005034405"  # Previous log directory
GCCB_CHECKPOINT="log/psnet12/respointnet2_dp_fi_df_fc1_max_gc_cb_loss_20251003071346/best.pth"  # 80 epochs, 44.31% mIoU
GCCB_LOG_DIR="log/psnet12/respointnet2_dp_fi_df_fc1_max_gc_cb_loss_20251003071346"  # Previous log directory
COMBINED_LOG_DIR_BASE="log/psnet12/respointnet2_dp_fi_df_fc1_max_combined_loss"

# Color codes for pretty printing
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${CYAN}"
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to run a single experiment
run_experiment() {
    local exp_id=$1
    local exp_name=$2
    local config_file=$3
    local extra_args=$4
    local checkpoint_path=$5  # New parameter for checkpoint path
    local log_dir=$6  # New parameter for log directory (for resuming)
    
    local port=$((BASE_PORT + EXPERIMENT_PORT_OFFSET))
    EXPERIMENT_PORT_OFFSET=$((EXPERIMENT_PORT_OFFSET + 1))
    
    print_header "EXPERIMENT ${exp_id}: ${exp_name}"
    print_info "Config: ${config_file}"
    print_info "Resuming from: ${checkpoint_path}"
    print_info "Log directory: ${log_dir}"
    print_info "Extra Args: ${extra_args}"
    print_info "Port: ${port}"
    print_info "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local start_time=$(date +%s)
    
    # Ensure we're in the project root directory
    cd /media/yinchao/code/ResPointNet2
    
    # If resuming from a checkpoint into a NEW directory for a new experiment type (e.g. combined),
    # we don't need to rename the old log.txt from the source directory.
    # The new log_dir won't have a log.txt yet.
    if [ -n "$log_dir" ] && [ -f "${log_dir}/log.txt" ]; then
        if [[ "$exp_name" != *"combined"* ]]; then # Avoid renaming for combined loss to keep original log
            print_info "Renaming existing log.txt to log_old.txt in ${log_dir}"
            mv "${log_dir}/log.txt" "${log_dir}/log_old.txt"
        fi
    fi
    
    # Run training using torchrun
    torchrun --master_port ${port} \
        --nproc_per_node ${NUM_GPUS} \
        function/train_psnet_dist.py \
        --dataset_name ${DATASET_NAME} \
        --cfg cfgs/${DATASET_NAME}/${config_file} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        --load_path ${checkpoint_path} \
        --log_dir ${log_dir} \
        ${extra_args} 2>&1 | tee "experiment_logs/paper_sprint_exp${exp_id}_${exp_name}.log"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    print_success "Experiment ${exp_id} completed!"
    print_info "Duration: ${hours}h ${minutes}m"
    print_info "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# =============================================================================
# Experiment Definitions
# =============================================================================

# Experiment 1: Local Density-Aware CB Loss (LD-CB) - Continue training
# Resume from LD-CB checkpoint: 58 epochs, 42.77% mIoU, patience 20→50
exp1_ld_cb() {
    run_experiment 1 "ld_cb_loss" \
        "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
        "--cb_loss_type ld --ld_density_weight 0.4" \
        "${LDCB_CHECKPOINT}" \
        "${LDCB_LOG_DIR}"
}

# Experiment 2: Geometric Context-Aware CB Loss (GC-CB) - Continue training
# Resume from GC-CB checkpoint: 80 epochs, 44.31% mIoU, patience 20→50
# Paper states k=16 is used for experiments. 
exp2_gc_cb() {
    run_experiment 2 "gc_cb_loss" \
        "respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml" \
        "--cb_loss_type gc --gc_context_k 32" \
        "${GCCB_CHECKPOINT}" \
        "${GCCB_LOG_DIR}"
}

# Experiment 3: Combined LD-CB and GC-CB Loss
# Start from GC-CB checkpoint (better performance: 44.31% vs 42.77%)
# Writes to a new log directory.
exp3_combined_loss() {
    local timestamp=$(date '+%Y%m%d%H%M%S')
    local combined_log_dir="${COMBINED_LOG_DIR_BASE}_${timestamp}"
    run_experiment 3 "combined_ld_gc_cb_loss" \
        "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
        "--cb_loss_type combined --ld_density_radius 0.5 --gc_context_k 32 --gc_context_weight 0.5" \
        "${GCCB_CHECKPOINT}" \
        "${combined_log_dir}"
}

# Experiment 4: Ablation on LD-CB radius 'r'
# Test different radius values starting from LD-CB checkpoint
# Aligned with paper Table: 0.25, 1.0, 2.0 (default is 0.5)
exp4_ld_cb_radius_ablation() {
    print_header "ABLATION STUDY: LD-CB Density Radius 'r' (default=0.4m)"
    local radii=(0.2 0.6 0.8)
    for radius in "${radii[@]}"; do
        local radius_str="${radius//./_}"
        local exp_id="4_ld_radius_${radius_str}"
        print_info "Testing density_radius = ${radius}m"
        
        run_experiment "${exp_id}" "ld_cb_radius_${radius_str}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --ld_density_radius ${radius}" \
            "${LDCB_CHECKPOINT}" \
            "${LDCB_LOG_DIR}/ablation_r_${radius_str}" # Use sub-directory for ablation logs
    done
}

# Experiment 5: Ablation on GC-CB neighborhood size 'k'
# Test different k values starting from GC-CB checkpoint
# Main experiments now use k=32 as default.
exp5_gc_cb_k_ablation() {
    print_header "ABLATION STUDY: GC-CB Neighborhood Size 'k' (default=16)"
    local k_values=(8 32 64)
    for k in "${k_values[@]}"; do
        local exp_id="5_gc_k_${k}"
        print_info "Testing context_k = ${k}"
        
        run_experiment "${exp_id}" "gc_cb_k${k}" \
            "respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml" \
            "--cb_loss_type gc --gc_context_k ${k}" \
            "${GCCB_CHECKPOINT}" \
            "${GCCB_LOG_DIR}/ablation_k_${k}" # Use sub-directory for ablation logs
    done
}


# =============================================================================
# Main Execution Logic
# =============================================================================

main() {
    local exp_to_run=${1:-"all"}
    
    print_header "PAPER SPRINT CORE EXPERIMENTS (ActionPlan-CB.md)"
    
    # Verify checkpoints exist
    if [ ! -f "${LDCB_CHECKPOINT}" ]; then
        print_error "LD-CB checkpoint not found at: ${LDCB_CHECKPOINT}"
        exit 1
    fi
    if [ ! -f "${GCCB_CHECKPOINT}" ]; then
        print_error "GC-CB checkpoint not found at: ${GCCB_CHECKPOINT}"
        exit 1
    fi
    
    print_info "Checkpoint Status:"
    print_info "  LD-CB: 58 epochs, 42.77% mIoU → ${LDCB_CHECKPOINT}"
    print_info "  GC-CB: 80 epochs, 44.31% mIoU → ${GCCB_CHECKPOINT}"
    print_info "Early stopping patience: 20 → ${EARLY_STOPPING_PATIENCE} epochs"
    print_info "Experiments align with ActionPlan-CB.md (Days 1-4)"
    echo ""
    
    mkdir -p experiment_logs
    
    if [ "$exp_to_run" = "all" ]; then
        print_warning "Running ALL paper sprint experiments sequentially."
        sleep 5
        exp1_ld_cb
        exp2_gc_cb
        exp3_combined_loss
        exp4_ld_cb_radius_ablation
        exp5_gc_cb_k_ablation
    else
        for exp_num in $@; do
            case ${exp_num} in
                1) exp1_ld_cb ;;
                2) exp2_gc_cb ;;
                3) exp3_combined_loss ;;
                4) exp4_ld_cb_radius_ablation ;;
                5) exp5_gc_cb_k_ablation ;;
                *)
                    print_error "Invalid experiment number: ${exp_num}"
                    print_info "Valid options: 1-5"
                    continue
                    ;;
            esac
        done
    fi
    
    print_header "ALL REQUESTED SPRINT EXPERIMENTS COMPLETED!"
    print_success "Check experiment_logs/ for individual experiment logs."
}

# =============================================================================
# Run Script
# =============================================================================

if [ $# -eq 0 ]; then
    # Interactive mode - show menu
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║           PAPER SPRINT EXPERIMENTS - EXPERIMENT MENU                       ║"
    echo "║                  (Aligned with ActionPlan-CB.md)                           ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "CORE EXPERIMENTS (Days 1-2):"
    echo "  1  - LD-CB Loss (continue from checkpoint, r=0.4, patience=50) ⭐"
    echo "  2  - GC-CB Loss (continue from checkpoint, k=16, patience=50) ⭐"
    echo "  3  - Combined LD-CB + GC-CB Loss (from GC-CB checkpoint) ⭐"
    echo ""
    echo "ABLATION STUDIES (Day 4):"
    echo "  4  - LD-CB: Density Radius Ablation (r=0.2, 0.6, 0.8; default=0.4)"
    echo "  5  - GC-CB: Neighborhood Size Ablation (k=8, 32, 64; default=16)"
    echo ""
    echo "BATCH OPTIONS:"
    echo "  all - Run ALL experiments (WARNING: Very long!)"
    echo ""
    echo "MULTIPLE EXPERIMENTS:"
    echo "  You can run multiple: bash $0 1 2 3"
    echo ""
    echo -n "Enter experiment number(s) (or 'all'): "
    read exp_choice
    main ${exp_choice}
else
    # Command line arguments provided
    main "$@"
fi
