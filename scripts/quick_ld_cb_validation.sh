#!/bin/bash

################################################################################
# Quick LD-CB Loss Validation Experiments
#
# Purpose: Rapidly test key hypotheses to diagnose LD-CB underperformance
# Expected runtime: 6-12 hours (3 experiments, ~2-4h each with early stopping)
#
# Run: bash scripts/quick_ld_cb_validation.sh
################################################################################

set -e

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
    echo "GPU with sufficient memory found. Starting paper experiments..."
else
    echo "No GPU with 19GB+ memory available. Will retry in 3 hours."
    sleep 10800  # Wait for 3 hours 
    
    # Check again after waiting
    if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
        echo "GPU with sufficient memory now available. Starting experiments..."
    else
        echo "Still no GPU with sufficient memory. Will retry in 3 hours."
        sleep 10800  # Wait for 3 hours
        if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
            echo "GPU with sufficient memory now available. Starting experiments..."
        else
            sleep 10800  # Wait for 3 hours
            if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
                echo "GPU with sufficient memory now available. Starting experiments..."
            else
                echo "Still no GPU with sufficient memory. Exiting."
                exit 1
            fi
        fi
    fi
fi

NUM_GPUS=2
DATASET_NAME='psnet12'
BASE_PORT=12360
EARLY_STOPPING_PATIENCE=20

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "=============================================================================="
echo "QUICK LD-CB VALIDATION EXPERIMENTS"
echo "=============================================================================="
echo -e "${NC}"

cd /media/yinchao/code/ResPointNet2
mkdir -p experiment_logs

# =============================================================================
# Experiment A: LD-CB with density_weight=0.0 (Density DISABLED)
# Expected: Should match Strong CB performance (~54% mIoU)
# Purpose: Verify that density modulation is causing the poor performance
# =============================================================================

echo -e "${CYAN}[Exp A] LD-CB with density_weight=0.0 (Density DISABLED)${NC}"
echo "Expected: ~54% mIoU (should match Strong CB)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

torchrun --master_port $((BASE_PORT + 0)) \
    --nproc_per_node ${NUM_GPUS} \
    function/train_psnet_dist.py \
    --dataset_name ${DATASET_NAME} \
    --cfg cfgs/${DATASET_NAME}/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
    --cb_loss_type ld \
    --ld_density_weight 0.0 \
    --cb_gamma 3.0 \
    --cb_alpha 2.0 \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    2>&1 | tee "experiment_logs/quick_validation_A_density_weight_0.0.log"

echo -e "${GREEN}[Exp A] Completed!${NC}"
echo ""

# =============================================================================
# Experiment B: LD-CB with density_weight=0.2 (Gentle Density)
# Expected: 50-52% mIoU (better than current 46%)
# Purpose: Test if gentle density modulation helps
# =============================================================================

echo -e "${CYAN}[Exp B] LD-CB with density_weight=0.2 (Gentle Density)${NC}"
echo "Expected: 50-52% mIoU"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

torchrun --master_port $((BASE_PORT + 1)) \
    --nproc_per_node ${NUM_GPUS} \
    function/train_psnet_dist.py \
    --dataset_name ${DATASET_NAME} \
    --cfg cfgs/${DATASET_NAME}/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
    --cb_loss_type ld \
    --ld_density_weight 0.2 \
    --cb_gamma 3.0 \
    --cb_alpha 2.0 \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    2>&1 | tee "experiment_logs/quick_validation_B_density_weight_0.2.log"

echo -e "${GREEN}[Exp B] Completed!${NC}"
echo ""

# =============================================================================
# Experiment C: LD-CB with density_weight=0.4 (Medium Density)
# Expected: 51-53% mIoU
# Purpose: Find optimal density weight
# =============================================================================

echo -e "${CYAN}[Exp C] LD-CB with density_weight=0.4 (Medium Density)${NC}"
echo "Expected: 51-53% mIoU"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

torchrun --master_port $((BASE_PORT + 2)) \
    --nproc_per_node ${NUM_GPUS} \
    function/train_psnet_dist.py \
    --dataset_name ${DATASET_NAME} \
    --cfg cfgs/${DATASET_NAME}/respointnet2_dp_fi_df_fc1_max_ld_cb_loss.yaml \
    --cb_loss_type ld \
    --ld_density_weight 0.4 \
    --cb_gamma 3.0 \
    --cb_alpha 2.0 \
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    2>&1 | tee "experiment_logs/quick_validation_C_density_weight_0.4.log"

echo -e "${GREEN}[Exp C] Completed!${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo -e "${CYAN}"
echo "=============================================================================="
echo "ALL QUICK VALIDATION EXPERIMENTS COMPLETED!"
echo "=============================================================================="
echo -e "${NC}"

echo "Results:"
echo "  Exp A (density_weight=0.0): See experiment_logs/quick_validation_A_density_weight_0.0.log"
echo "  Exp B (density_weight=0.2): See experiment_logs/quick_validation_B_density_weight_0.2.log"
echo "  Exp C (density_weight=0.4): See experiment_logs/quick_validation_C_density_weight_0.4.log"
echo ""

echo "Next Steps:"
echo "  1. Extract best mIoU from each log:"
echo "     grep 'New best mIoU' experiment_logs/quick_validation_*.log"
echo ""
echo "  2. Compare results with baseline:"
echo "     - Strong CB: 54.09% mIoU"
echo "     - Current LD-CB (weight=1.0): 46.01% mIoU"
echo ""
echo "  3. If Exp A (~54%) >> Exp B (~52%) >> current (46%):"
echo "     → Density modulation is too aggressive, use gentle weight (0.2-0.4)"
echo ""
echo "  4. Run full ablation studies with promising weight range:"
echo "     bash scripts/comprehensive_CB_loss_experiments.sh 6"
echo ""

echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"

