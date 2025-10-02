#!/bin/bash

################################################################################
# Smart GPU Scheduler for CB Loss Experiments
#
# This script monitors GPU resources and automatically runs experiments
# when sufficient memory becomes available. It's designed for limited GPU
# resources and can run experiments in the background.
#
# Features:
# - Monitors GPU memory every 2-4 hours
# - Automatically starts experiments when >19GB memory available
# - Queues experiments and runs them sequentially
# - Saves progress and can resume from interruptions
# - Sends notifications when experiments complete
#
# Usage:
#   bash scripts/gpu_scheduler.sh [start_experiment_number]
#
# Author: Research Team
# Date: 2025-10-01
################################################################################

set -e

# =============================================================================
# Configuration
# =============================================================================

# GPU Requirements
REQUIRED_MEMORY_GB=19
CHECK_INTERVAL_HOURS=3  # Check every 3 hours
MAX_WAIT_DAYS=7         # Maximum wait time before giving up

# Experiment Configuration
NUM_GPUS=2
DATASET_NAME='psnet12'
LOG_ROOT="log/${DATASET_NAME}"
BASE_PORT=12350
EARLY_STOPPING_PATIENCE=20

# Scheduler State Files
SCHEDULER_DIR="gpu_scheduler_state"
QUEUE_FILE="${SCHEDULER_DIR}/experiment_queue.txt"
PROGRESS_FILE="${SCHEDULER_DIR}/progress.txt"
LOCK_FILE="${SCHEDULER_DIR}/scheduler.lock"

# Color codes for output
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

# Function to check if scheduler is already running
check_scheduler_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid=$(cat "$LOCK_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_warning "GPU Scheduler is already running (PID: $pid)"
            print_info "To stop it: kill $pid"
            exit 1
        else
            print_info "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
}

# Function to create scheduler lock
create_lock() {
    echo $$ > "$LOCK_FILE"
}

# Function to remove scheduler lock
remove_lock() {
    rm -f "$LOCK_FILE"
}

# Function to check GPU memory availability
check_gpu_memory() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. Cannot check GPU memory."
        return 1
    fi
    
    # Get available memory for each GPU
    local gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    local max_memory=0
    
    for mem in $gpu_info; do
        if [ "$mem" -gt "$max_memory" ]; then
            max_memory=$mem
        fi
    done
    
    local memory_gb=$((max_memory / 1024))
    print_info "Maximum available GPU memory: ${memory_gb}GB"
    
    if [ "$max_memory" -ge $((REQUIRED_MEMORY_GB * 1024)) ]; then
        return 0  # Sufficient memory available
    else
        return 1  # Insufficient memory
    fi
}

# Function to get GPU utilization
get_gpu_utilization() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1
    else
        echo "0"
    fi
}

# Function to check if any training is currently running
check_running_training() {
    # Check for running Python processes with training scripts
    local training_processes=$(pgrep -f "train_psnet_dist.py" | wc -l)
    if [ "$training_processes" -gt 0 ]; then
        return 0  # Training is running
    else
        return 1  # No training running
    fi
}

# Function to initialize scheduler state
init_scheduler() {
    mkdir -p "$SCHEDULER_DIR"
    
    # Create experiment queue if it doesn't exist
    if [ ! -f "$QUEUE_FILE" ]; then
        cat > "$QUEUE_FILE" << EOF
# GPU Scheduler Experiment Queue
# Format: experiment_number:experiment_name:priority:status
# Status: pending, running, completed, failed

# Core experiments (CRITICAL)
1:baseline_ce_loss:high:pending
2:standard_cb_loss:high:completed
3:ld_cb_loss_primary:high:pending
4:gc_cb_loss_secondary:high:pending

# LD-CB ablations (HIGH PRIORITY)
5:ld_cb_radius_ablation:high:pending
6:ld_cb_weight_ablation:high:pending

# GC-CB ablations (MEDIUM PRIORITY)
7:gc_cb_k_ablation:medium:pending
8:gc_cb_weight_ablation:medium:pending

# Additional analyses (LOW PRIORITY)
9:beta_ablation:low:pending
10:gamma_ablation:low:pending
11:lr_sensitivity:low:pending
12:seed_robustness:low:pending
13:gc_entropy_temp:low:pending
14:efficiency_test:low:pending
15:gradient_clipping:low:pending
EOF
    fi
    
    # Create progress file if it doesn't exist
    if [ ! -f "$PROGRESS_FILE" ]; then
        cat > "$PROGRESS_FILE" << EOF
# GPU Scheduler Progress Log
# Format: timestamp:experiment:status:message

$(date '+%Y-%m-%d %H:%M:%S'):scheduler:started:GPU Scheduler initialized
EOF
    fi
}

# Function to get next experiment from queue
get_next_experiment() {
    # Get highest priority pending experiment
    local next_exp=$(grep ":pending" "$QUEUE_FILE" | grep ":high:" | head -1)
    if [ -z "$next_exp" ]; then
        next_exp=$(grep ":pending" "$QUEUE_FILE" | grep ":medium:" | head -1)
    fi
    if [ -z "$next_exp" ]; then
        next_exp=$(grep ":pending" "$QUEUE_FILE" | grep ":low:" | head -1)
    fi
    
    if [ -n "$next_exp" ]; then
        echo "$next_exp" | cut -d':' -f1
    else
        echo ""
    fi
}

# Function to update experiment status
update_experiment_status() {
    local exp_num=$1
    local status=$2
    local message=$3
    
    # Update queue file
    sed -i "s/^${exp_num}:.*:${status}/&/" "$QUEUE_FILE"
    sed -i "s/:pending:/:${status}:/" "$QUEUE_FILE"
    
    # Add to progress log
    echo "$(date '+%Y-%m-%d %H:%M:%S'):exp${exp_num}:${status}:${message}" >> "$PROGRESS_FILE"
}

# Function to run a single experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local port=$((BASE_PORT + exp_num))
    
    print_header "STARTING EXPERIMENT ${exp_num}: ${exp_name}"
    print_info "Port: ${port}"
    print_info "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Update status to running
    update_experiment_status "$exp_num" "running" "Started experiment"
    
    local start_time=$(date +%s)
    local log_file="gpu_scheduler_logs/exp${exp_num}_${exp_name}.log"
    mkdir -p gpu_scheduler_logs
    
    # Run the experiment
    if python -m torch.distributed.launch --master_port ${port} \
        --nproc_per_node ${NUM_GPUS} \
        function/train_psnet_dist.py \
        --dataset_name ${DATASET_NAME} \
        --cfg cfgs/${DATASET_NAME}/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
        2>&1 | tee "$log_file"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        
        print_success "Experiment ${exp_num} completed successfully!"
        print_info "Duration: ${hours}h ${minutes}m"
        update_experiment_status "$exp_num" "completed" "Successfully completed in ${hours}h ${minutes}m"
        
        # Send notification (if configured)
        send_notification "Experiment ${exp_num} (${exp_name}) completed successfully in ${hours}h ${minutes}m"
        
    else
        print_error "Experiment ${exp_num} failed!"
        update_experiment_status "$exp_num" "failed" "Experiment failed - check logs"
        send_notification "Experiment ${exp_num} (${exp_name}) failed - check logs"
    fi
}

# Function to send notification (customize as needed)
send_notification() {
    local message="$1"
    
    # Option 1: Email notification (if configured)
    # echo "$message" | mail -s "GPU Scheduler Update" your_email@domain.com
    
    # Option 2: Desktop notification (if available)
    if command -v notify-send &> /dev/null; then
        notify-send "GPU Scheduler" "$message"
    fi
    
    # Option 3: Log to file
    echo "$(date '+%Y-%m-%d %H:%M:%S'):NOTIFICATION:$message" >> "$PROGRESS_FILE"
    
    print_info "Notification: $message"
}

# Function to monitor GPU and run experiments
monitor_and_run() {
    local start_time=$(date +%s)
    local max_wait_seconds=$((MAX_WAIT_DAYS * 24 * 3600))
    local check_interval_seconds=$((CHECK_INTERVAL_HOURS * 3600))
    
    print_header "GPU SCHEDULER STARTED"
    print_info "Required memory: ${REQUIRED_MEMORY_GB}GB"
    print_info "Check interval: ${CHECK_INTERVAL_HOURS} hours"
    print_info "Max wait time: ${MAX_WAIT_DAYS} days"
    print_info "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    while true; do
        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))
        
        # Check if we've exceeded maximum wait time
        if [ $elapsed_time -gt $max_wait_seconds ]; then
            print_error "Maximum wait time (${MAX_WAIT_DAYS} days) exceeded. Exiting."
            break
        fi
        
        print_info "Checking GPU resources... (Elapsed: $((elapsed_time / 3600))h)"
        
        # Check if any training is currently running
        if check_running_training; then
            print_info "Training already in progress. Waiting..."
            sleep 300  # Wait 5 minutes before checking again
            continue
        fi
        
        # Check GPU memory
        if check_gpu_memory; then
            print_success "Sufficient GPU memory available!"
            
            # Get next experiment
            local next_exp=$(get_next_experiment)
            if [ -n "$next_exp" ]; then
                print_info "Starting experiment ${next_exp}"
                run_experiment "$next_exp" "experiment_${next_exp}"
            else
                print_success "All experiments completed!"
                break
            fi
        else
            print_warning "Insufficient GPU memory. Waiting ${CHECK_INTERVAL_HOURS} hours..."
            
            # Show current GPU status
            if command -v nvidia-smi &> /dev/null; then
                print_info "Current GPU status:"
                nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv
            fi
            
            # Wait for next check
            sleep $check_interval_seconds
        fi
    done
    
    print_header "GPU SCHEDULER COMPLETED"
    print_info "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    print_info "Total runtime: $((elapsed_time / 3600)) hours"
}

# Function to show scheduler status
show_status() {
    print_header "GPU SCHEDULER STATUS"
    
    if [ -f "$PROGRESS_FILE" ]; then
        print_info "Recent activity:"
        tail -10 "$PROGRESS_FILE"
    fi
    
    if [ -f "$QUEUE_FILE" ]; then
        print_info "Experiment queue:"
        cat "$QUEUE_FILE" | grep -v "^#"
    fi
    
    print_info "GPU status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv
    else
        print_warning "nvidia-smi not available"
    fi
}

# Function to reset scheduler
reset_scheduler() {
    print_warning "Resetting scheduler state..."
    rm -rf "$SCHEDULER_DIR"
    print_success "Scheduler state reset. Run again to reinitialize."
}

# Function to show help
show_help() {
    cat << EOF
GPU Scheduler for CB Loss Experiments

USAGE:
    bash scripts/gpu_scheduler.sh [COMMAND]

COMMANDS:
    start           Start the GPU scheduler (default)
    status          Show current scheduler status
    reset           Reset scheduler state
    help            Show this help message

EXAMPLES:
    # Start scheduler (will monitor and run experiments automatically)
    bash scripts/gpu_scheduler.sh start
    
    # Check current status
    bash scripts/gpu_scheduler.sh status
    
    # Reset if something went wrong
    bash scripts/gpu_scheduler.sh reset

FEATURES:
    - Monitors GPU memory every ${CHECK_INTERVAL_HOURS} hours
    - Automatically runs experiments when >${REQUIRED_MEMORY_GB}GB available
    - Queues experiments by priority
    - Saves progress and can resume
    - Sends notifications on completion/failure
    - Handles interruptions gracefully

CONFIGURATION:
    Required memory: ${REQUIRED_MEMORY_GB}GB
    Check interval: ${CHECK_INTERVAL_HOURS} hours
    Max wait time: ${MAX_WAIT_DAYS} days
    Number of GPUs: ${NUM_GPUS}

EOF
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local command=${1:-"start"}
    
    case "$command" in
        "start")
            check_scheduler_lock
            create_lock
            init_scheduler
            monitor_and_run
            remove_lock
            ;;
        "status")
            show_status
            ;;
        "reset")
            reset_scheduler
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'print_warning "Scheduler interrupted. Cleaning up..."; remove_lock; exit 0' INT TERM

# Run main function
main "$@"
