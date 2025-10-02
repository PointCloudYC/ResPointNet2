#!/bin/bash

################################################################################
# Efficient GPU Resource Manager for CB Loss Experiments
#
# This script efficiently manages GPU resources by:
# 1. Running experiments with adaptive batch sizes based on available memory
# 2. Using smaller models/configs when memory is limited
# 3. Running multiple short experiments in sequence when memory is available
# 4. Automatically adjusting to available resources
#
# Usage:
#   bash scripts/efficient_gpu_runner.sh [mode]
#   Modes: adaptive, quick, full
#
# Author: Research Team
# Date: 2025-10-01
################################################################################

set -e

# =============================================================================
# Configuration
# =============================================================================

# GPU Memory Thresholds (in GB)
LOW_MEMORY_THRESHOLD=8
MEDIUM_MEMORY_THRESHOLD=12
HIGH_MEMORY_THRESHOLD=19

# Experiment Configurations by Memory Level
declare -A BATCH_SIZES=(
    ["low"]="2"
    ["medium"]="4" 
    ["high"]="6"
)

declare -A NUM_POINTS=(
    ["low"]="10000"
    ["medium"]="15000"
    ["high"]="15000"
)

declare -A EPOCHS=(
    ["low"]="300"
    ["medium"]="400"
    ["high"]="600"
)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

# Function to get available GPU memory
get_available_memory() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "0"
        return
    fi
    
    local max_memory=0
    local gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    
    for mem in $gpu_info; do
        if [ "$mem" -gt "$max_memory" ]; then
            max_memory=$mem
        fi
    done
    
    echo $((max_memory / 1024))  # Convert to GB
}

# Function to determine memory level
get_memory_level() {
    local available_memory=$(get_available_memory)
    
    if [ "$available_memory" -ge "$HIGH_MEMORY_THRESHOLD" ]; then
        echo "high"
    elif [ "$available_memory" -ge "$MEDIUM_MEMORY_THRESHOLD" ]; then
        echo "medium"
    elif [ "$available_memory" -ge "$LOW_MEMORY_THRESHOLD" ]; then
        echo "low"
    else
        echo "insufficient"
    fi
}

# Function to create adaptive config
create_adaptive_config() {
    local memory_level=$1
    local base_config=$2
    local output_config=$3
    
    local batch_size=${BATCH_SIZES[$memory_level]}
    local num_points=${NUM_POINTS[$memory_level]}
    local epochs=${EPOCHS[$memory_level]}
    
    print_info "Creating adaptive config for ${memory_level} memory level:"
    print_info "  Batch size: ${batch_size}"
    print_info "  Num points: ${num_points}"
    print_info "  Epochs: ${epochs}"
    
    # Copy base config and modify
    cp "$base_config" "$output_config"
    
    # Update parameters using sed
    sed -i "s/batch_size: [0-9]*/batch_size: ${batch_size}/" "$output_config"
    sed -i "s/num_points: [0-9]*/num_points: ${num_points}/" "$output_config"
    sed -i "s/epochs: [0-9]*/epochs: ${epochs}/" "$output_config"
    
    # Add adaptive config note
    echo "" >> "$output_config"
    echo "# Adaptive configuration for ${memory_level} memory level" >> "$output_config"
    echo "# Generated on: $(date)" >> "$output_config"
}

# Function to run quick experiment (for testing)
run_quick_experiment() {
    local exp_name=$1
    local config_file=$2
    local port=$3
    
    print_header "QUICK EXPERIMENT: ${exp_name}"
    print_info "Config: ${config_file}"
    print_info "Port: ${port}"
    print_info "Duration: ~30 minutes (10 epochs)"
    
    local start_time=$(date +%s)
    
    # Run with reduced epochs for quick testing
    python -m torch.distributed.launch --master_port ${port} \
        --nproc_per_node 2 \
        function/train_psnet_dist.py \
        --dataset_name psnet12 \
        --cfg "${config_file}" \
        --epochs 10 \
        --early_stopping_patience 5 \
        --print_freq 5 \
        --val_freq 2 \
        --save_freq 5
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    
    print_success "Quick experiment completed in ${minutes} minutes"
}

# Function to run full experiment
run_full_experiment() {
    local exp_name=$1
    local config_file=$2
    local port=$3
    
    print_header "FULL EXPERIMENT: ${exp_name}"
    print_info "Config: ${config_file}"
    print_info "Port: ${port}"
    print_info "Duration: 2-3 days"
    
    local start_time=$(date +%s)
    
    python -m torch.distributed.launch --master_port ${port} \
        --nproc_per_node 2 \
        function/train_psnet_dist.py \
        --dataset_name psnet12 \
        --cfg "${config_file}" \
        --early_stopping_patience 20
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    print_success "Full experiment completed in ${hours}h ${minutes}m"
}

# Function to run adaptive experiment
run_adaptive_experiment() {
    local exp_name=$1
    local base_config=$2
    local port=$3
    
    local memory_level=$(get_memory_level)
    local available_memory=$(get_available_memory)
    
    print_header "ADAPTIVE EXPERIMENT: ${exp_name}"
    print_info "Available memory: ${available_memory}GB"
    print_info "Memory level: ${memory_level}"
    
    if [ "$memory_level" = "insufficient" ]; then
        print_error "Insufficient GPU memory (${available_memory}GB < ${LOW_MEMORY_THRESHOLD}GB)"
        print_info "Waiting for more memory or try quick mode"
        return 1
    fi
    
    # Create adaptive config
    local adaptive_config="cfgs/psnet12/adaptive_${exp_name}_${memory_level}.yaml"
    create_adaptive_config "$memory_level" "$base_config" "$adaptive_config"
    
    # Run experiment
    run_full_experiment "$exp_name" "$adaptive_config" "$port"
}

# Function to run multiple quick experiments
run_quick_suite() {
    print_header "QUICK EXPERIMENT SUITE"
    print_info "Running multiple short experiments to test all methods"
    
    local experiments=(
        "baseline:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml"
        "ld_cb:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml"
        "gc_cb:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml"
    )
    
    local port=12350
    
    for exp in "${experiments[@]}"; do
        local exp_name=$(echo "$exp" | cut -d':' -f1)
        local config_file=$(echo "$exp" | cut -d':' -f2)
        
        print_info "Starting quick experiment: ${exp_name}"
        run_quick_experiment "${exp_name}_quick" "$config_file" "$port"
        
        port=$((port + 1))
        
        # Wait a bit between experiments
        sleep 60
    done
    
    print_success "Quick experiment suite completed!"
}

# Function to run core experiments adaptively
run_core_experiments() {
    print_header "CORE EXPERIMENTS (ADAPTIVE)"
    print_info "Running the most important experiments with adaptive resource usage"
    
    local experiments=(
        "baseline:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml"
        "ld_cb:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml"
        "gc_cb:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml"
    )
    
    local port=12350
    
    for exp in "${experiments[@]}"; do
        local exp_name=$(echo "$exp" | cut -d':' -f1)
        local config_file=$(echo "$exp" | cut -d':' -f2)
        
        print_info "Starting adaptive experiment: ${exp_name}"
        
        if run_adaptive_experiment "$exp_name" "$config_file" "$port"; then
            print_success "Experiment ${exp_name} completed successfully"
        else
            print_warning "Experiment ${exp_name} skipped due to insufficient resources"
        fi
        
        port=$((port + 1))
        
        # Wait between experiments
        sleep 300  # 5 minutes
    done
}

# Function to monitor and run when resources are available
monitor_and_run() {
    print_header "RESOURCE MONITORING MODE"
    print_info "Will check GPU resources every 30 minutes and run experiments when available"
    
    local check_interval=1800  # 30 minutes
    local max_checks=48         # Check for 24 hours
    
    for ((i=1; i<=max_checks; i++)); do
        local available_memory=$(get_available_memory)
        local memory_level=$(get_memory_level)
        
        print_info "Check ${i}/${max_checks}: Available memory: ${available_memory}GB (Level: ${memory_level})"
        
        if [ "$memory_level" != "insufficient" ]; then
            print_success "Sufficient resources available! Starting experiments..."
            run_core_experiments
            break
        else
            print_warning "Insufficient resources. Waiting ${check_interval} seconds..."
            sleep $check_interval
        fi
    done
}

# Function to show current GPU status
show_gpu_status() {
    print_header "CURRENT GPU STATUS"
    
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu,temperature.gpu --format=csv
        
        local available_memory=$(get_available_memory)
        local memory_level=$(get_memory_level)
        
        print_info "Available memory: ${available_memory}GB"
        print_info "Memory level: ${memory_level}"
        
        case "$memory_level" in
            "high")
                print_success "High memory available - can run full experiments"
                ;;
            "medium")
                print_warning "Medium memory - will use adaptive settings"
                ;;
            "low")
                print_warning "Low memory - will use reduced settings"
                ;;
            "insufficient")
                print_error "Insufficient memory - cannot run experiments"
                ;;
        esac
    else
        print_error "nvidia-smi not available"
    fi
}

# Function to show help
show_help() {
    cat << EOF
Efficient GPU Resource Manager for CB Loss Experiments

USAGE:
    bash scripts/efficient_gpu_runner.sh [MODE]

MODES:
    adaptive         Run experiments with adaptive resource usage (RECOMMENDED)
    quick           Run quick test experiments (10 epochs each)
    full            Run full experiments (600 epochs)
    monitor         Monitor resources and run when available
    status          Show current GPU status
    help            Show this help

EXAMPLES:
    # Run with adaptive resource usage (best for limited GPU)
    bash scripts/efficient_gpu_runner.sh adaptive
    
    # Run quick tests to verify everything works
    bash scripts/efficient_gpu_runner.sh quick
    
    # Monitor and run when resources become available
    bash scripts/efficient_gpu_runner.sh monitor
    
    # Check current GPU status
    bash scripts/efficient_gpu_runner.sh status

ADAPTIVE MODE FEATURES:
    - Automatically adjusts batch size based on available memory
    - Uses smaller models when memory is limited
    - Runs experiments in sequence to maximize resource usage
    - Can handle 8GB+ GPUs efficiently

MEMORY REQUIREMENTS:
    - High (19GB+): Full experiments with batch_size=6
    - Medium (12-19GB): Adaptive experiments with batch_size=4
    - Low (8-12GB): Reduced experiments with batch_size=2
    - Insufficient (<8GB): Cannot run experiments

EOF
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local mode=${1:-"adaptive"}
    
    case "$mode" in
        "adaptive")
            run_core_experiments
            ;;
        "quick")
            run_quick_suite
            ;;
        "full")
            print_warning "Full mode - ensure you have sufficient GPU memory (19GB+)"
            run_core_experiments
            ;;
        "monitor")
            monitor_and_run
            ;;
        "status")
            show_gpu_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown mode: $mode"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
