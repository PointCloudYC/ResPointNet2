#!/bin/bash

################################################################################
# GPU Resource Daemon for CB Loss Experiments
#
# A simple daemon that monitors GPU resources and automatically runs
# experiments when sufficient memory becomes available.
#
# Features:
# - Runs in background
# - Checks GPU every 2-4 hours
# - Automatically starts experiments when >19GB available
# - Can be started/stopped easily
# - Logs all activity
#
# Usage:
#   bash scripts/gpu_daemon.sh start    # Start daemon
#   bash scripts/gpu_daemon.sh stop     # Stop daemon
#   bash scripts/gpu_daemon.sh status   # Check status
#   bash scripts/gpu_daemon.sh logs    # View logs
#
# Author: Research Team
# Date: 2025-10-01
################################################################################

set -e

# =============================================================================
# Configuration
# =============================================================================

DAEMON_DIR="gpu_daemon"
PID_FILE="${DAEMON_DIR}/daemon.pid"
LOG_FILE="${DAEMON_DIR}/daemon.log"
STATE_FILE="${DAEMON_DIR}/state.txt"

# GPU Requirements
REQUIRED_MEMORY_GB=19
CHECK_INTERVAL_HOURS=3
MAX_WAIT_DAYS=7

# Experiment Configuration
NUM_GPUS=2
BASE_PORT=12350

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

# Function to log messages
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" | tee -a "$LOG_FILE"
}

# Function to check if daemon is running
is_daemon_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Daemon is running
        else
            # Stale PID file
            rm -f "$PID_FILE"
            return 1
        fi
    else
        return 1  # No PID file
    fi
}

# Function to get daemon PID
get_daemon_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

# Function to check GPU memory
check_gpu_memory() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "ERROR: nvidia-smi not found"
        return 1
    fi
    
    local gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    local max_memory=0
    
    for mem in $gpu_info; do
        if [ "$mem" -gt "$max_memory" ]; then
            max_memory=$mem
        fi
    done
    
    local memory_gb=$((max_memory / 1024))
    log_message "Maximum available GPU memory: ${memory_gb}GB"
    
    if [ "$max_memory" -ge $((REQUIRED_MEMORY_GB * 1024)) ]; then
        return 0  # Sufficient memory
    else
        return 1  # Insufficient memory
    fi
}

# Function to check if training is running
is_training_running() {
    local training_processes=$(pgrep -f "train_psnet_dist.py" | wc -l)
    [ "$training_processes" -gt 0 ]
}

# Function to run experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local config_file=$3
    local port=$((BASE_PORT + exp_num))
    
    log_message "Starting experiment ${exp_num}: ${exp_name}"
    
    # Update state
    echo "running:${exp_num}:${exp_name}:$(date '+%Y-%m-%d %H:%M:%S')" > "$STATE_FILE"
    
    local start_time=$(date +%s)
    local log_file="${DAEMON_DIR}/exp${exp_num}_${exp_name}.log"
    
    # Run experiment
    if python -m torch.distributed.launch --master_port ${port} \
        --nproc_per_node ${NUM_GPUS} \
        function/train_psnet_dist.py \
        --dataset_name psnet12 \
        --cfg "${config_file}" \
        --early_stopping_patience 20 \
        2>&1 | tee "$log_file"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        
        log_message "Experiment ${exp_num} completed successfully in ${hours}h ${minutes}m"
        echo "completed:${exp_num}:${exp_name}:$(date '+%Y-%m-%d %H:%M:%S')" > "$STATE_FILE"
        
    else
        log_message "Experiment ${exp_num} failed"
        echo "failed:${exp_num}:${exp_name}:$(date '+%Y-%m-%d %H:%M:%S')" > "$STATE_FILE"
    fi
}

# Function to get next experiment
get_next_experiment() {
    # Simple queue - you can modify this to be more sophisticated
    local experiments=(
        "1:baseline:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml"
        "3:ld_cb:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml"
        "4:gc_cb:cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_gc_cb_loss.yaml"
    )
    
    # Check which experiments have been completed
    for exp in "${experiments[@]}"; do
        local exp_num=$(echo "$exp" | cut -d':' -f1)
        local exp_name=$(echo "$exp" | cut -d':' -f2)
        local config_file=$(echo "$exp" | cut -d':' -f3)
        
        # Check if this experiment is already completed
        if [ -f "$STATE_FILE" ]; then
            local state=$(cat "$STATE_FILE")
            if echo "$state" | grep -q "completed:${exp_num}:"; then
                continue  # Skip completed experiments
            fi
        fi
        
        # Return this experiment
        echo "$exp"
        return
    done
    
    # No more experiments
    echo ""
}

# Function to start daemon
start_daemon() {
    if is_daemon_running; then
        local pid=$(get_daemon_pid)
        print_warning "Daemon is already running (PID: $pid)"
        return 1
    fi
    
    # Create daemon directory
    mkdir -p "$DAEMON_DIR"
    
    # Start daemon in background
    nohup bash -c '
        # Daemon main loop
        while true; do
            # Check if we should stop
            if [ ! -f "'"$PID_FILE"'" ]; then
                log_message "Daemon stopping - PID file removed"
                exit 0
            fi
            
            # Check if training is already running
            if is_training_running; then
                log_message "Training already in progress, waiting..."
                sleep 1800  # Wait 30 minutes
                continue
            fi
            
            # Check GPU memory
            if check_gpu_memory; then
                log_message "Sufficient GPU memory available, starting experiment"
                
                # Get next experiment
                local next_exp=$(get_next_experiment)
                if [ -n "$next_exp" ]; then
                    local exp_num=$(echo "$next_exp" | cut -d":" -f1)
                    local exp_name=$(echo "$next_exp" | cut -d":" -f2)
                    local config_file=$(echo "$next_exp" | cut -d":" -f3)
                    
                    run_experiment "$exp_num" "$exp_name" "$config_file"
                else
                    log_message "All experiments completed!"
                    break
                fi
            else
                log_message "Insufficient GPU memory, waiting '"$CHECK_INTERVAL_HOURS"' hours..."
                sleep $((CHECK_INTERVAL_HOURS * 3600))
            fi
        done
        
        log_message "Daemon completed all experiments"
    ' > /dev/null 2>&1 &
    
    local daemon_pid=$!
    echo "$daemon_pid" > "$PID_FILE"
    
    print_success "Daemon started (PID: $daemon_pid)"
    log_message "GPU Daemon started (PID: $daemon_pid)"
    
    # Show initial status
    show_status
}

# Function to stop daemon
stop_daemon() {
    if ! is_daemon_running; then
        print_warning "Daemon is not running"
        return 1
    fi
    
    local pid=$(get_daemon_pid)
    print_info "Stopping daemon (PID: $pid)..."
    
    # Remove PID file to signal daemon to stop
    rm -f "$PID_FILE"
    
    # Wait for daemon to stop gracefully
    sleep 5
    
    # Force kill if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        print_warning "Force stopping daemon..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
    fi
    
    print_success "Daemon stopped"
    log_message "GPU Daemon stopped"
}

# Function to show daemon status
show_status() {
    print_header "GPU DAEMON STATUS"
    
    if is_daemon_running; then
        local pid=$(get_daemon_pid)
        print_success "Daemon is running (PID: $pid)"
        
        if [ -f "$STATE_FILE" ]; then
            local state=$(cat "$STATE_FILE")
            print_info "Current state: $state"
        fi
    else
        print_warning "Daemon is not running"
    fi
    
    print_info "GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv
    else
        print_error "nvidia-smi not available"
    fi
    
    print_info "Recent daemon activity:"
    if [ -f "$LOG_FILE" ]; then
        tail -10 "$LOG_FILE"
    else
        print_warning "No log file found"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        print_header "DAEMON LOGS"
        cat "$LOG_FILE"
    else
        print_warning "No log file found"
    fi
}

# Function to reset daemon
reset_daemon() {
    print_warning "Resetting daemon state..."
    
    if is_daemon_running; then
        stop_daemon
    fi
    
    rm -rf "$DAEMON_DIR"
    print_success "Daemon state reset"
}

# Function to show help
show_help() {
    cat << EOF
GPU Resource Daemon for CB Loss Experiments

USAGE:
    bash scripts/gpu_daemon.sh [COMMAND]

COMMANDS:
    start           Start the GPU daemon
    stop            Stop the GPU daemon
    status          Show daemon status
    logs            Show daemon logs
    reset           Reset daemon state
    help            Show this help

EXAMPLES:
    # Start daemon (will run in background)
    bash scripts/gpu_daemon.sh start
    
    # Check if daemon is running
    bash scripts/gpu_daemon.sh status
    
    # Stop daemon
    bash scripts/gpu_daemon.sh stop
    
    # View logs
    bash scripts/gpu_daemon.sh logs

FEATURES:
    - Runs in background
    - Checks GPU every ${CHECK_INTERVAL_HOURS} hours
    - Automatically runs experiments when >${REQUIRED_MEMORY_GB}GB available
    - Logs all activity
    - Can be started/stopped easily

REQUIREMENTS:
    - ${REQUIRED_MEMORY_GB}GB+ GPU memory
    - nvidia-smi available
    - 2 GPUs recommended

EOF
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local command=${1:-"help"}
    
    case "$command" in
        "start")
            start_daemon
            ;;
        "stop")
            stop_daemon
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "reset")
            reset_daemon
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

# Run main function
main "$@"
