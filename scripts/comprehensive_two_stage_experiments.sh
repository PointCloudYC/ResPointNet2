#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# ============================================================================
# COMPREHENSIVE TWO-STAGE TRAINING EXPERIMENTS
# ============================================================================
# Tests all combinations of CE/CB loss in Stage 1 and Stage 2
# Uses pre-trained Stage 1 checkpoints to save time
#
# Experiments:
#   1. S1 CE → S2 CE (baseline)
#   2. S1 CB → S2 CE (recommended: strong S1 + clean S2)
#   3. S1 CB → S2 CB (test: complementary or redundant?)
#   4. S1 CE → S2 CB (test: can CB help weak S1?)
#
# Additional ablations:
#   5. S1 CB → S2 CE with lower LR (0.0001)
#   6. S1 CB → S2 CE with unfrozen encoder
# ============================================================================

# --- Configuration ---
num_gpus=2
dataset_name='psnet12'
# config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage"
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_warmup"
log_root="log/${dataset_name}"
master_port_base=12350

# Pre-trained Stage 1 checkpoints
cb_stage1_checkpoint="log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth"  # 54.088% mIoU
ce_stage1_checkpoint="log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054/best.pth"  # 52.479% mIoU

# Training parameters
stage1_epochs=80  # Doesn't matter since we skip Stage 1
stage2_epochs=40
stage2_warmup_epochs=10
sampling_temperature=2.0
patience=20

echo "============================================================================"
echo "COMPREHENSIVE TWO-STAGE TRAINING EXPERIMENTS"
echo "============================================================================"
echo ""
echo "Using pre-trained Stage 1 checkpoints:"
echo "  CB Stage 1: ${cb_stage1_checkpoint} (54.088% mIoU)"
echo "  CE Stage 1: ${ce_stage1_checkpoint} (52.479% mIoU)"
echo ""
echo "Stage 2 configuration:"
echo "  Epochs: ${stage2_epochs}"
echo "  Balanced sampling: ENABLED (class-aware center selection)"
echo "  Encoder frozen: YES (default)"
echo ""
echo "============================================================================"
echo ""

# Check if checkpoints exist
if [ ! -f "${cb_stage1_checkpoint}" ]; then
    echo "❌ ERROR: CB Stage 1 checkpoint not found at ${cb_stage1_checkpoint}"
    echo "Please train Stage 1 with CB loss first or update the path"
    exit 1
fi

if [ ! -f "${ce_stage1_checkpoint}" ]; then
    echo "❌ ERROR: CE Stage 1 checkpoint not found at ${ce_stage1_checkpoint}"
    echo "Please train Stage 1 with CE loss first or update the path"
    exit 1
fi

echo "✅ All required checkpoints found"
echo ""

# Function to run an experiment
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local stage1_ckpt=$3
    local use_cb_s1=$4
    local use_cb_s2=$5
    local stage2_lr=$6
    local stage2_epochs_local=$7
    local freeze_encoder=$8
    local description=$9
    
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT ${exp_num}: ${exp_name}"
    echo "========================================================================"
    echo "${description}"
    echo ""
    echo "Configuration:"
    echo "  Stage 1: $([ "$use_cb_s1" = "true" ] && echo "CB loss (54.088%)" || echo "CE loss (52.479%)")"
    echo "  Stage 2: $([ "$use_cb_s2" = "true" ] && echo "CB loss" || echo "CE loss")"
    echo "  Stage 2 LR: ${stage2_lr}"
    echo "  Stage 2 Epochs: ${stage2_epochs_local}"
    echo "  Encoder frozen: $([ "$freeze_encoder" = "true" ] && echo "YES" || echo "NO")"
    echo ""
    
    # Create experiment directory
    timestamp=$(date +%Y%m%d%H%M%S)
    exp_dir="${log_root}/${config_name}_exp${exp_num}_${exp_name}_${timestamp}"
    mkdir -p "${exp_dir}/stage1"
    mkdir -p "${exp_dir}/stage2"
    
    # Copy the appropriate checkpoint
    echo "Setting up Stage 1 checkpoint..."
    cp "${stage1_ckpt}" "${exp_dir}/stage1/stage1_best.pth"
    echo "✓ Checkpoint copied to: ${exp_dir}/stage1/stage1_best.pth"
    echo ""
    
    # Build command
    cmd="python -m torch.distributed.launch"
    cmd="${cmd} --master_port $((master_port_base + exp_num))"
    cmd="${cmd} --nproc_per_node ${num_gpus}"
    cmd="${cmd} function/train_psnet_two_stage_dist.py"
    cmd="${cmd} --dataset_name ${dataset_name}"
    cmd="${cmd} --cfg cfgs/${dataset_name}/${config_name}.yaml"
    cmd="${cmd} --log_dir ${exp_dir}"
    cmd="${cmd} --stage1_epochs ${stage1_epochs}"
    cmd="${cmd} --stage2_epochs ${stage2_epochs_local}"
    cmd="${cmd} --stage2_lr ${stage2_lr}"
    cmd="${cmd} --stage2_warmup_epochs ${stage2_warmup_epochs}"
    cmd="${cmd} --sampling_temperature ${sampling_temperature}"
    cmd="${cmd} --early_stopping_patience ${patience}"
    
    # Add CB loss flags
    [ "$use_cb_s1" = "true" ] && cmd="${cmd} --use_cb_loss_stage1"
    [ "$use_cb_s2" = "true" ] && cmd="${cmd} --use_cb_loss_stage2"
    
    # Add freeze encoder flag
    [ "$freeze_encoder" = "true" ] && cmd="${cmd} --freeze_encoder"
    
    echo "Running command:"
    echo "${cmd}"
    echo ""
    
    # Run experiment
    start_time=$(date +%s)
    eval "time ${cmd}"
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "✅ Experiment ${exp_num} completed in ${duration}s"
    echo "📁 Results: ${exp_dir}"
    echo "========================================================================"
    echo ""
    
    # Save experiment info
    cat > "${exp_dir}/experiment_info.txt" << EOF
Experiment ${exp_num}: ${exp_name}
========================================
${description}

Configuration:
  Stage 1 Loss: $([ "$use_cb_s1" = "true" ] && echo "CB" || echo "CE")
  Stage 2 Loss: $([ "$use_cb_s2" = "true" ] && echo "CB" || echo "CE")
  Stage 1 Checkpoint: ${stage1_ckpt}
  Stage 2 LR: ${stage2_lr}
  Stage 2 Epochs: ${stage2_epochs_local}
  Stage 2 Warmup Epochs: ${stage2_warmup_epochs}
  Sampling Temperature: ${sampling_temperature}
  Encoder Frozen: $([ "$freeze_encoder" = "true" ] && echo "YES" || echo "NO")
  
Timestamp: ${timestamp}
Duration: ${duration}s
Results Directory: ${exp_dir}
EOF
    
    # Short pause between experiments
    sleep 3
}

# ============================================================================
# CORE EXPERIMENTS: Test all 4 combinations
# ============================================================================

echo "🔬 CORE EXPERIMENTS: Testing all CE/CB loss combinations"
echo ""

# Experiment 1: S1 CE → S2 CE (baseline)
run_experiment 1 \
    "ce_s1_ce_s2" \
    "${ce_stage1_checkpoint}" \
    "false" \
    "false" \
    "0.0002" \
    "20" \
    "true" \
    "Baseline: CE loss throughout (consistent but weaker S1 foundation)"

# Experiment 2: S1 CB → S2 CE (recommended)
run_experiment 2 \
    "cb_s1_ce_s2" \
    "${cb_stage1_checkpoint}" \
    "true" \
    "false" \
    "0.0002" \
    "20" \
    "true" \
    "RECOMMENDED: Strong CB S1 foundation + clean CE S2 optimization"

# Experiment 3: S1 CB → S2 CB (test redundancy)
run_experiment 3 \
    "cb_s1_cb_s2" \
    "${cb_stage1_checkpoint}" \
    "true" \
    "true" \
    "0.0002" \
    "20" \
    "true" \
    "Test: Does CB loss in S2 help or hurt with balanced sampling?"

# Experiment 4: S1 CE → S2 CB (test if CB can rescue weak S1)
run_experiment 4 \
    "ce_s1_cb_s2" \
    "${ce_stage1_checkpoint}" \
    "false" \
    "true" \
    "0.0002" \
    "20" \
    "true" \
    "Test: Can CB loss in S2 compensate for weaker CE S1 foundation?"

# ============================================================================
# ABLATION STUDIES: Stage 2 hyperparameters
# ============================================================================

echo ""
echo "🔬 ABLATION STUDIES: Stage 2 hyperparameters"
echo ""

# Experiment 5: Lower LR (original setting from old experiments)
run_experiment 5 \
    "cb_s1_ce_s2_lr0001" \
    "${cb_stage1_checkpoint}" \
    "true" \
    "false" \
    "0.0001" \
    "20" \
    "true" \
    "Ablation: Lower LR (0.0001) - original setting that worked"

# Experiment 6: Higher LR
run_experiment 7 \
    "cb_s1_ce_s2_lr0003" \
    "${cb_stage1_checkpoint}" \
    "true" \
    "false" \
    "0.0003" \
    "20" \
    "true" \
    "Ablation: Higher LR (0.0003) - test faster convergence"

# Experiment 8: Best of both worlds (more epochs + moderate LR)
run_experiment 8 \
    "cb_s1_ce_s2_ep40_lr00015" \
    "${cb_stage1_checkpoint}" \
    "true" \
    "false" \
    "0.00015" \
    "40" \
    "true" \
    "Ablation: 40 epochs + LR 0.00015 (balanced approach)"

# ============================================================================
# ADVANCED EXPERIMENT: Unfrozen encoder
# ============================================================================

echo ""
echo "🔬 ADVANCED EXPERIMENT: Unfrozen encoder"
echo ""

# Experiment 9: S1 CB → S2 CE with unfrozen encoder
run_experiment 9 \
    "cb_s1_ce_s2_unfrozen" \
    "${cb_stage1_checkpoint}" \
    "true" \
    "false" \
    "0.0001" \
    "20" \
    "false" \
    "Advanced: Fine-tune entire network (encoder NOT frozen)"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "============================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================================"
echo ""
echo "Results summary:"
echo ""
echo "CORE EXPERIMENTS (Loss Combinations):"
echo "  1. CE→CE:  ${log_root}/${config_name}_exp1_*"
echo "  2. CB→CE:  ${log_root}/${config_name}_exp2_* (RECOMMENDED)"
echo "  3. CB→CB:  ${log_root}/${config_name}_exp3_*"
echo "  4. CE→CB:  ${log_root}/${config_name}_exp4_*"
echo ""
echo "ABLATION STUDIES (Hyperparameters):"
echo "  5. LR=0.0001:     ${log_root}/${config_name}_exp5_*"
# echo "  6. Epochs=40:     ${log_root}/${config_name}_exp6_*"
echo "  6. LR=0.0003:     ${log_root}/${config_name}_exp7_*"
echo "  7. Best balance:  ${log_root}/${config_name}_exp8_*"
echo ""
echo "ADVANCED:"
echo "  8. Unfrozen encoder: ${log_root}/${config_name}_exp9_*"
echo ""
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. Compare final mIoU across all experiments"
echo "2. Focus on per-class mIoU for tail classes (mechanical, furniture, etc.)"
echo "3. Analyze training curves in TensorBoard"
echo "4. Check if new class-aware sampler actually helps tail classes"
echo ""
echo "Key questions to answer:"
echo "  - Does CB→CE (Exp 2) beat the old results?"
echo "  - Is CB loss helpful in S2 (Exp 3) or redundant?"
echo "  - What's the optimal LR and epochs for S2?"
echo "  - Does unfrozen encoder (Exp 9) improve results?"
echo ""
echo "============================================================================"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_two_stage_results.py --log_dir ${log_root}"
echo ""
