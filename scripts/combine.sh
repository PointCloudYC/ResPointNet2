#!/bin/bash

# Function to check GPU memory availability
check_gpu_memory() {
    # Get available memory for each GPU
    gpu_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    
    for mem in $gpu_info; do
        if [ "$mem" -ge 19000 ]; then
            return 0  # At least one GPU has 19GB+ free memory
        fi
    done
    
    return 1  # No GPU with enough memory
}

# Check if GPUs are available and have enough memory
if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
    echo "GPU with sufficient memory found. Starting experiments..."
else
    echo "No GPU with 19GB+ memory available. Will retry in 2 hours."
    sleep 7200  # Wait for 2 hours (7200 seconds)
    
    # Check again after waiting
    if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
        echo "GPU with sufficient memory now available. Starting experiments..."
    else
        echo "Still no GPU with sufficient memory. Exiting."
        exit 1
    fi
fi

echo "run the following commands to conduct several experiments"

echo "conduct strong cb loss training"
bash scripts/run_psnet12_cb_loss_strong_exp.sh
echo "complete the strong cb loss training"

echo "conduct two-stage training"
bash scripts/run_psnet12_two_stage_exp.sh
echo "complete the two-stage training"