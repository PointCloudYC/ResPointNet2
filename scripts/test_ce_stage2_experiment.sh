#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Test CE Loss in Stage 2 with Different Stage 1 Foundations
# Hypothesis: CE loss in Stage 2 might be better since balanced sampling already handles imbalance

# --- Configuration ---
num_gpus=2
dataset_name='psnet12'
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_ce_s2"
log_root="log/${dataset_name}"
master_port_train=12349

# Stage 1 checkpoints for comparison
cb_stage1_checkpoint="log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth"  # 54.088% mIoU
ce_stage1_checkpoint="log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054/best.pth"  # 52.479% mIoU

# Test parameters
stage1_epochs=80
stage2_epochs=20
stage2_lr=0.0001  # This showed best results in your experiments
patience=20

echo "=================================================================="
echo "EXPERIMENT: CE Loss in Stage 2 with Different Stage 1 Foundations"
echo "=================================================================="
echo "Hypothesis: CE loss in Stage 2 might be better since balanced"
echo "           sampling already handles class imbalance"
echo ""
echo "Test 2: CB Stage 1 (54.088%) → CE Stage 2"
echo "Test 3: CE Stage 1 (52.479%) → CE Stage 2"
echo ""
echo "Stage 2 learning rate: ${stage2_lr}"
echo "=================================================================="

# Check if both Stage 1 checkpoints exist
if [ ! -f "${cb_stage1_checkpoint}" ]; then
    echo "❌ ERROR: CB Stage 1 checkpoint not found at ${cb_stage1_checkpoint}"
    exit 1
fi

if [ ! -f "${ce_stage1_checkpoint}" ]; then
    echo "❌ ERROR: CE Stage 1 checkpoint not found at ${ce_stage1_checkpoint}"
    exit 1
fi

echo "✓ Found CB Stage 1 checkpoint (54.088% mIoU)"
echo "✓ Found CE Stage 1 checkpoint (52.479% mIoU)"
echo ""

# Test 2: CB Stage 1 → CE Stage 2
echo "🧪 TEST 2: CB Stage 1 (54.088%) → CE Stage 2 with Balanced Sampling"
echo "===================================================================="
echo "This tests whether CE loss in Stage 2 works well with strong Stage 1 features"
echo ""

# Create experiment directory with timestamp for Test 2
timestamp=$(date +%Y%m%d%H%M%S)
exp_dir_test2="${log_root}/${config_name}_cb_s1_ce_s2_${timestamp}"
mkdir -p "${exp_dir_test2}/stage1"
mkdir -p "${exp_dir_test2}/stage2"

# Copy the CB pretrained checkpoint as stage1_best.pth
echo "Copying CB pretrained checkpoint to stage1 directory..."
cp "${cb_stage1_checkpoint}" "${exp_dir_test2}/stage1/stage1_best.pth"
echo "Stage 1 checkpoint ready at: ${exp_dir_test2}/stage1/stage1_best.pth"

time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_two_stage_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --log_dir "${exp_dir_test2}" \
    --stage1_epochs ${stage1_epochs} \
    --stage2_epochs ${stage2_epochs} \
    --stage2_lr ${stage2_lr} \
    --use_cb_loss_stage1 \
    --freeze_encoder \
    --early_stopping_patience ${patience}

echo "✅ Test 2 completed"
echo "Results saved in: ${exp_dir_test2}"
echo ""

# Wait a moment
sleep 5

# Test 3: CE Stage 1 → CE Stage 2  
echo "🧪 TEST 3: CE Stage 1 (52.479%) → CE Stage 2 with Balanced Sampling"
echo "===================================================================="
echo "This tests whether CE loss throughout can compete with CB→CE approach"
echo ""

# Create experiment directory with timestamp for Test 3
timestamp=$(date +%Y%m%d%H%M%S)
exp_dir_test3="${log_root}/${config_name}_ce_s1_ce_s2_${timestamp}"
mkdir -p "${exp_dir_test3}/stage1"
mkdir -p "${exp_dir_test3}/stage2"

# Copy the CE pretrained checkpoint as stage1_best.pth
echo "Copying CE pretrained checkpoint to stage1 directory..."
cp "${ce_stage1_checkpoint}" "${exp_dir_test3}/stage1/stage1_best.pth"
echo "Stage 1 checkpoint ready at: ${exp_dir_test3}/stage1/stage1_best.pth"

time python -m torch.distributed.launch --master_port $((master_port_train + 1)) \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_two_stage_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --log_dir "${exp_dir_test3}" \
    --stage1_epochs ${stage1_epochs} \
    --stage2_epochs ${stage2_epochs} \
    --stage2_lr ${stage2_lr} \
    --freeze_encoder \
    --early_stopping_patience ${patience}

echo "✅ Test 3 completed"
echo "Results saved in: ${exp_dir_test3}"
echo ""

echo "=================================================================="
echo "EXPERIMENT COMPLETED"
echo "=================================================================="
echo "Compare the results from both tests:"
echo ""
echo "Test 2 (CB S1 → CE S2): ${exp_dir_test2}"
echo "                        Expected: ~54.1-54.5% (strong foundation + clean Stage 2)"
echo ""
echo "Test 3 (CE S1 → CE S2): ${exp_dir_test3}"  
echo "                        Expected: ~52.5-53.0% (weaker foundation but consistent)"
echo ""
echo "Key Questions:"
echo "1. Does CE loss in Stage 2 maintain the benefits of CB Stage 1?"
echo "2. Can CE→CE approach compete with CB→CE?"
echo "3. Is the Stage 1 foundation more important than Stage 2 loss choice?"
echo "=================================================================="
