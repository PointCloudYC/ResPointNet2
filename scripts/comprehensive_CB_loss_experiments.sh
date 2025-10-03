#!/bin/bash

################################################################################
# Comprehensive CB Loss Experiments for PSNet12
#
# This script runs a comprehensive suite of experiments for the CB loss paper:
# 1. Baseline (Standard CE Loss - already done)
# 2. Standard CB Loss (Strong version - already done)
# 3. Local Density-Aware CB Loss (LD-CB) - PRIMARY CONTRIBUTION
# 4. Geometric Context-Aware CB Loss (GC-CB) - SECONDARY CONTRIBUTION
# 5. Ablations on LD-CB hyperparameters
# 6. Ablations on GC-CB hyperparameters
# 7. Analysis experiments
#
# Usage:
#   bash scripts/comprehensive_CB_loss_experiments.sh [experiment_number]
#   - No argument: runs all experiments sequentially
#   - With argument: runs specific experiment (1-15)
#
# Author: Research Team
# Date: 2025-10-01
################################################################################

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

NUM_GPUS=2
DATASET_NAME='psnet12'
LOG_ROOT="log/${DATASET_NAME}"
BASE_PORT=12350  # Starting port, will increment for each experiment
EARLY_STOPPING_PATIENCE=20

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
    local exp_num=$1
    local exp_name=$2
    local config_file=$3
    local extra_args=$4
    local port=$((BASE_PORT + exp_num))
    
    print_header "EXPERIMENT ${exp_num}: ${exp_name}"
    print_info "Config: ${config_file}"
    print_info "Extra Args: ${extra_args}"
    print_info "Port: ${port}"
    print_info "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local start_time=$(date +%s)
    
    # Ensure we're in the project root directory
    cd /media/yinchao/code/ResPointNet2
    
    # Run training using torchrun (updated from deprecated torch.distributed.launch)
    torchrun --master_port ${port} \
        --nproc_per_node ${NUM_GPUS} \
        function/train_psnet_dist.py \
        --dataset_name ${DATASET_NAME} \
        --cfg cfgs/${DATASET_NAME}/${config_file} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        ${extra_args} 2>&1 | tee "experiment_logs/exp${exp_num}_${exp_name}.log"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    print_success "Experiment ${exp_num} completed!"
    print_info "Duration: ${hours}h ${minutes}m"
    print_info "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

# =============================================================================
# Experiment Definitions
# =============================================================================

# Experiment 1: Baseline (Standard CE Loss)
exp1_baseline() {
    run_experiment 1 "baseline_ce_loss" \
        "respointnet2_dp_fi_df_fc1_max.yaml" \
        "--cb_loss_type none"
}

# Experiment 2: Standard CB Loss (Strong version)
exp2_standard_cb() {
    print_header "EXPERIMENT 2: Standard CB Loss (Strong)"
    print_warning "This experiment was already completed!"
    print_info "Log directory: log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005"
    print_info "Skipping re-run. If you want to re-run, delete the log directory first."
    echo ""
}

# Experiment 3: Local Density-Aware CB Loss (PRIMARY CONTRIBUTION)
exp3_ld_cb() {
    run_experiment 3 "ld_cb_loss_primary" \
        "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
        "--cb_loss_type ld"
}

# Experiment 4: Geometric Context-Aware CB Loss (SECONDARY CONTRIBUTION)
exp4_gc_cb() {
    run_experiment 4 "gc_cb_loss_secondary" \
        "respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml" \
        "--cb_loss_type gc"
}

# =============================================================================
# Ablation Studies: LD-CB Loss Hyperparameters
# =============================================================================

# Experiment 5: LD-CB with different density_radius values
exp5_ld_cb_radius_ablation() {
    print_header "ABLATION STUDY: LD-CB Density Radius"
    
    # Test different radius values: 0.3, 0.5 (default), 0.7, 1.0
    local radii=(0.3 0.5 0.7 1.0)
    
    for radius in "${radii[@]}"; do
        # Replace dots with underscores for valid bash variable names
        local radius_str="${radius//./_}"
        local exp_id="5_ld_radius_${radius_str}"
        print_info "Testing density_radius = ${radius}m"
        
        run_experiment ${exp_id} "ld_cb_radius${radius}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --ld_density_radius ${radius}"
    done
}

# Experiment 6: LD-CB with different density_weight values
exp6_ld_cb_weight_ablation() {
    print_header "ABLATION STUDY: LD-CB Density Weight"
    
    # Test different weight values: 0.0 (disabled), 0.5, 1.0 (default), 1.5
    local weights=(0.0 0.5 1.0 1.5)
    
    for weight in "${weights[@]}"; do
        # Replace dots with underscores for valid bash variable names
        local weight_str="${weight//./_}"
        local exp_id="6_ld_weight_${weight_str}"
        print_info "Testing density_weight = ${weight}"
        
        run_experiment ${exp_id} "ld_cb_weight${weight}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --ld_density_weight ${weight}"
    done
}

# =============================================================================
# Ablation Studies: GC-CB Loss Hyperparameters
# =============================================================================

# Experiment 7: GC-CB with different context_k values
exp7_gc_cb_k_ablation() {
    print_header "ABLATION STUDY: GC-CB Context K (Number of Neighbors)"
    
    # Test different k values: 8, 16 (default), 32, 64
    local k_values=(8 16 32 64)
    
    for k in "${k_values[@]}"; do
        local exp_id="7_gc_k_${k}"
        print_info "Testing context_k = ${k}"
        
        run_experiment ${exp_id} "gc_cb_k${k}" \
            "respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml" \
            "--cb_loss_type gc --gc_context_k ${k}"
    done
}

# Experiment 8: GC-CB with different context_weight values
exp8_gc_cb_weight_ablation() {
    print_header "ABLATION STUDY: GC-CB Context Weight (Lambda)"
    
    # Test different weight values: 0.0 (disabled), 0.5, 1.0 (default), 2.0
    local weights=(0.0 0.5 1.0 2.0)
    
    for weight in "${weights[@]}"; do
        local exp_id="8_gc_weight_${weight}"
        print_info "Testing context_weight = ${weight}"
        
        run_experiment ${exp_id} "gc_cb_weight${weight}" \
            "respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml" \
            "--cb_loss_type gc --gc_context_weight ${weight}"
    done
}

# =============================================================================
# Additional Important Experiments
# =============================================================================

# Experiment 9: Beta parameter comparison (CB Loss hyperparameter)
exp9_beta_ablation() {
    print_header "ABLATION STUDY: CB Beta Parameter"
    
    # Test different beta values: 0.99, 0.999, 0.9999 (default), 0.99999
    local betas=(0.99 0.999 0.9999 0.99999)
    
    for beta in "${betas[@]}"; do
        # Replace dots with underscores for valid bash variable names
        local beta_str="${beta//./_}"
        local exp_id="9_beta_${beta_str}"
        print_info "Testing cb_beta = ${beta}"
        
        # Test with standard CB loss
        run_experiment "${exp_id}_standard" "standard_cb_beta${beta}" \
            "respointnet2_dp_fi_df_fc1_max_cb_loss.yaml" \
            "--cb_beta ${beta}"
        
        # Test with LD-CB loss
        run_experiment "${exp_id}_ld" "ld_cb_beta${beta}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --cb_beta ${beta}"
    done
}

# Experiment 10: Gamma parameter comparison (Focal loss hyperparameter)
exp10_gamma_ablation() {
    print_header "ABLATION STUDY: Focal Loss Gamma Parameter"
    
    # Test different gamma values: 0.0 (no focal), 1.0, 2.0 (default), 3.0
    local gammas=(0.0 1.0 2.0 3.0)
    
    for gamma in "${gammas[@]}"; do
        # Replace dots with underscores for valid bash variable names
        local gamma_str="${gamma//./_}"
        local exp_id="10_gamma_${gamma_str}"
        print_info "Testing cb_gamma = ${gamma}"
        
        run_experiment ${exp_id} "ld_cb_gamma${gamma}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --cb_gamma ${gamma}"
    done
}

# Experiment 11: Learning rate sensitivity for CB losses
exp11_lr_sensitivity() {
    print_header "ANALYSIS: Learning Rate Sensitivity"
    
    # Test different learning rates: 0.005, 0.01 (default), 0.02
    local lrs=(0.005 0.01 0.02)
    
    for lr in "${lrs[@]}"; do
        # Replace dots with underscores for valid bash variable names
        local lr_str="${lr//./_}"
        local exp_id="11_lr_${lr_str}"
        print_info "Testing base_learning_rate = ${lr}"
        
        run_experiment ${exp_id} "ld_cb_lr${lr}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --base_learning_rate ${lr}"
    done
}

# Experiment 12: Cross-validation with different random seeds
exp12_seed_robustness() {
    print_header "ANALYSIS: Random Seed Robustness"
    
    # Test with different random seeds
    local seeds=(0 42 123 456 789)
    
    for seed in "${seeds[@]}"; do
        local exp_id="12_seed_${seed}"
        print_info "Testing with random seed = ${seed}"
        
        run_experiment ${exp_id} "ld_cb_seed${seed}" \
            "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
            "--cb_loss_type ld --rng_seed ${seed}"
    done
}

# Experiment 13: LD-CB with entropy temperature ablation
exp13_gc_entropy_temp() {
    print_header "ABLATION STUDY: GC-CB Entropy Temperature"
    
    # Test different temperature values: 0.5, 1.0 (default), 2.0
    local temps=(0.5 1.0 2.0)
    
    for temp in "${temps[@]}"; do
        # Replace dots with underscores for valid bash variable names
        local temp_str="${temp//./_}"
        local exp_id="13_entropy_temp_${temp_str}"
        print_info "Testing entropy_temp = ${temp}"
        
        run_experiment ${exp_id} "gc_cb_temp${temp}" \
            "respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml" \
            "--cb_loss_type gc --gc_entropy_temp ${temp}"
    done
}

# Experiment 14: Training efficiency comparison (fewer epochs)
exp14_efficiency_test() {
    print_header "ANALYSIS: Training Efficiency (300 epochs)"
    
    print_info "Testing LD-CB with fewer epochs to measure sample efficiency"
    
    run_experiment "14_ld_300ep" "ld_cb_300epochs" \
        "respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml" \
        "--cb_loss_type ld --epochs 300"
    
    run_experiment "14_standard_300ep" "standard_cb_300epochs" \
        "respointnet2_dp_fi_df_fc1_max_cb_loss.yaml" \
        "--epochs 300"
}

# Experiment 15: Gradient clipping sensitivity
exp15_gradient_clipping() {
    print_header "ANALYSIS: Gradient Clipping (already implemented in code)"
    print_info "The code uses gradient clipping of 10.0 by default"
    print_info "This is already optimal based on prior work"
    print_info "Skipping this experiment to save time"
    echo ""
}

# =============================================================================
# Main Execution Logic
# =============================================================================

main() {
    local exp_to_run=${1:-"all"}
    
    print_header "COMPREHENSIVE CB LOSS EXPERIMENTS"
    print_info "Dataset: ${DATASET_NAME}"
    print_info "Number of GPUs: ${NUM_GPUS}"
    print_info "Early Stopping Patience: ${EARLY_STOPPING_PATIENCE}"
    print_info "Experiment to run: ${exp_to_run}"
    echo ""
    
    # Create experiment log directory
    mkdir -p experiment_logs
    
    # Handle multiple experiments or single experiment
    if [ "$exp_to_run" = "all" ]; then
        print_warning "Running ALL experiments sequentially. This will take a LONG time!"
        print_warning "Press Ctrl+C within 10 seconds to cancel..."
        sleep 10
        
        # Core experiments
        exp1_baseline
        exp2_standard_cb
        exp3_ld_cb
        exp4_gc_cb
        
        # LD-CB ablations (most important)
        exp5_ld_cb_radius_ablation
        exp6_ld_cb_weight_ablation
        
        # GC-CB ablations
        exp7_gc_cb_k_ablation
        exp8_gc_cb_weight_ablation
        
        # Additional analyses
        exp9_beta_ablation
        exp10_gamma_ablation
        exp11_lr_sensitivity
        exp12_seed_robustness
        exp13_gc_entropy_temp
        exp14_efficiency_test
        exp15_gradient_clipping
    else
        # Handle multiple experiments or single experiment
        for exp_num in $@; do
            case ${exp_num} in
                1) exp1_baseline ;;
                2) exp2_standard_cb ;;
                3) exp3_ld_cb ;;
                4) exp4_gc_cb ;;
                5) exp5_ld_cb_radius_ablation ;;
                6) exp6_ld_cb_weight_ablation ;;
                7) exp7_gc_cb_k_ablation ;;
                8) exp8_gc_cb_weight_ablation ;;
                9) exp9_beta_ablation ;;
                10) exp10_gamma_ablation ;;
                11) exp11_lr_sensitivity ;;
                12) exp12_seed_robustness ;;
                13) exp13_gc_entropy_temp ;;
                14) exp14_efficiency_test ;;
                15) exp15_gradient_clipping ;;
                *)
                    print_error "Invalid experiment number: ${exp_num}"
                    print_info "Valid options: 1-15"
                    continue
                    ;;
            esac
        done
    fi
    
    print_header "ALL REQUESTED EXPERIMENTS COMPLETED!"
    print_success "Check experiment_logs/ for individual experiment logs"
    print_success "Check ${LOG_ROOT}/ for model checkpoints and training logs"
}

# =============================================================================
# Run Script
# =============================================================================

# Parse command line arguments
if [ $# -eq 0 ]; then
    # Interactive mode - show menu
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║             COMPREHENSIVE CB LOSS EXPERIMENTS - EXPERIMENT MENU            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "CORE EXPERIMENTS:"
    echo "  1  - Baseline (Standard CE Loss)"
    echo "  2  - Standard CB Loss (already done - skip)"
    echo "  3  - Local Density-Aware CB Loss (PRIMARY CONTRIBUTION) ⭐"
    echo "  4  - Geometric Context-Aware CB Loss (SECONDARY CONTRIBUTION) ⭐"
    echo ""
    echo "ABLATION STUDIES - LD-CB:"
    echo "  5  - LD-CB: Density Radius Ablation (0.3, 0.5, 0.7, 1.0)"
    echo "  6  - LD-CB: Density Weight Ablation (0.0, 0.5, 1.0, 1.5)"
    echo ""
    echo "ABLATION STUDIES - GC-CB:"
    echo "  7  - GC-CB: Context K Ablation (8, 16, 32, 64)"
    echo "  8  - GC-CB: Context Weight Ablation (0.0, 0.5, 1.0, 2.0)"
    echo ""
    echo "ADDITIONAL ANALYSES:"
    echo "  9  - Beta Parameter Ablation (0.99, 0.999, 0.9999, 0.99999)"
    echo "  10 - Gamma Parameter Ablation (0.0, 1.0, 2.0, 3.0)"
    echo "  11 - Learning Rate Sensitivity (0.005, 0.01, 0.02)"
    echo "  12 - Random Seed Robustness (5 different seeds)"
    echo "  13 - GC-CB Entropy Temperature (0.5, 1.0, 2.0)"
    echo "  14 - Training Efficiency (300 epochs vs 600)"
    echo "  15 - Gradient Clipping (informational only)"
    echo ""
    echo "BATCH OPTIONS:"
    echo "  all - Run ALL experiments (WARNING: Very long!)"
    echo ""
    echo "MULTIPLE EXPERIMENTS:"
    echo "  You can run multiple experiments: bash script.sh 3 4 5"
    echo ""
    echo -n "Enter experiment number(s) (or 'all'): "
    read exp_choice
    main ${exp_choice}
else
    # Command line arguments provided - pass all arguments
    main "$@"
fi

