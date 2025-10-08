#!/bin/bash
#
# Test script to verify CB sampler fixes work correctly in distributed training
# This will run just a few epochs of Stage 2 to verify:
# 1. CB sampler statistics appear in logs
# 2. No distributed training errors occur
# 3. Class-balanced sampling is working
#

set -e  # Exit on error

# Configuration
CONFIG_FILE="cfgs/psnet12/respointnet2_dp_fi_df_fc1_max.yaml"
DATA_ROOT="data"
DATASET="psnet12"

# Find the most recent Stage 1 checkpoint
STAGE1_CHECKPOINT="log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_exp1_ce_s1_ce_s2_20250930174916/stage1/stage1_best.pth"

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at: $STAGE1_CHECKPOINT"
    echo "Please update STAGE1_CHECKPOINT path in this script"
    exit 1
fi

echo "=========================================="
echo "Testing CB Sampler Fixes"
echo "=========================================="
echo "Using Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo ""

# Create test log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG_DIR="log/${DATASET}/test_cb_sampler_fix_${TIMESTAMP}"
mkdir -p "$TEST_LOG_DIR"

echo "Test logs will be saved to: $TEST_LOG_DIR"
echo ""

# Test 1: S2 with CE loss (original failing config but with fixes)
echo "=========================================="
echo "Test 1: Stage 2 CE Loss (5 epochs)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29501 \
    function/train_psnet_two_stage_dist.py \
    --cfg "$CONFIG_FILE" \
    --data_root "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --log_dir "${TEST_LOG_DIR}/test1_ce" \
    --load_path "$STAGE1_CHECKPOINT" \
    --stage1_epochs 80 \
    --stage2_epochs 5 \
    --stage2_lr 0.0002 \
    --use_cb_loss_stage2 \
    --val_freq 5 \
    --save_freq 10

echo ""
echo "=========================================="
echo "Test 1 Complete!"
echo "=========================================="
echo ""
echo "Check the log file for:"
echo "  1. 'Class-Balanced Sampler Statistics' section"
echo "  2. 'Class selection counts for last epoch' after epoch 1 and 5"
echo "  3. Per-class sampling percentages"
echo ""
echo "Log file: ${TEST_LOG_DIR}/test1_ce/log.txt"
echo ""

# Test 2: S2 with CB loss (safety net)
echo "=========================================="
echo "Test 2: Stage 2 CB Loss (5 epochs)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29502 \
    function/train_psnet_two_stage_dist.py \
    --cfg "$CONFIG_FILE" \
    --data_root "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --log_dir "${TEST_LOG_DIR}/test2_cb" \
    --load_path "$STAGE1_CHECKPOINT" \
    --stage1_epochs 80 \
    --stage2_epochs 5 \
    --stage2_lr 0.0002 \
    --use_cb_loss_stage2 \
    --val_freq 5 \
    --save_freq 10

echo ""
echo "=========================================="
echo "Test 2 Complete!"
echo "=========================================="
echo ""

# Summary
echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="
echo ""
echo "Compare the results:"
echo "  Test 1 (CE):  ${TEST_LOG_DIR}/test1_ce/log.txt"
echo "  Test 2 (CB):  ${TEST_LOG_DIR}/test2_cb/log.txt"
echo ""
echo "What to check:"
echo "  ✓ CB sampler statistics appear in both logs"
echo "  ✓ Tail classes (7, 8, 10, 11) are being sampled"
echo "  ✓ No distributed training errors"
echo "  ✓ Training loss and validation loss are reasonable"
echo ""
