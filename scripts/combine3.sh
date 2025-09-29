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
    echo "GPU with sufficient memory found. Starting paper experiments..."
else
    echo "No GPU with 19GB+ memory available. Will retry in 4 hours."
    sleep 14400  # Wait for 4 hours 
    
    # Check again after waiting
    if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
        echo "GPU with sufficient memory now available. Starting experiments..."
    else
        echo "Still no GPU with sufficient memory. Will retry in 4 hours."
        sleep 14400  # Wait for 4 hours
        if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
            echo "GPU with sufficient memory now available. Starting experiments..."
        else
            sleep 14400  # Wait for 4 hours
            if command -v nvidia-smi &> /dev/null && check_gpu_memory; then
                echo "GPU with sufficient memory now available. Starting experiments..."
            else
                echo "Still no GPU with sufficient memory. Exiting."
                exit 1
            fi
        fi
    fi
fi

echo "=========================================="
echo "PAPER EXPERIMENTS - 3 EXPERIMENTS PLANNED"
echo "=========================================="
echo "1. Baseline ResPointNet++ with patience 50"
echo "2. Baseline + Strong CB Loss with patience 50"
echo "3. Baseline + Two-stage training with CB sampler and CB loss"
echo "=========================================="

# Experiment 1: Baseline ResPointNet++ with patience 50
echo "=========================================="
echo "EXPERIMENT 1: Baseline ResPointNet++ (patience 50)"
echo "Previous best: 49.5% mIoU (patience 20), 53.3% mIoU (600 epochs)"
echo "=========================================="
bash scripts/run_psne12_exp.sh 50
echo "EXPERIMENT 1 COMPLETED"
echo ""

# Experiment 2: Baseline + Strong CB Loss with patience 50
echo "=========================================="
echo "EXPERIMENT 2: Baseline + Strong CB Loss (patience 50)"
echo "Previous best: 50.072% mIoU @ epoch 119 (patience 20)"
echo "=========================================="
bash scripts/run_psnet12_cb_loss_strong_exp.sh 50
echo "EXPERIMENT 2 COMPLETED"
echo ""

# Experiment 3: Two-stage training using best checkpoint from Experiment 1 or 2 as Stage 1
echo "=========================================="
echo "EXPERIMENT 3: Two-stage training with pretrained Stage 1"
echo "Using best checkpoint from previous experiments as Stage 1"
echo "=========================================="

# Find the best checkpoint from the most recent experiments
echo "Looking for best checkpoints from recent experiments..."

# First, try to find the checkpoint from Experiment 1 (just completed)
exp1_checkpoint=$(ls -dt log/psnet12/respointnet2_dp_fi_df_fc1_max_* | head -1 | xargs -I {} find {} -name "best.pth" 2>/dev/null | head -1)

# If not found, try the checkpoint from Experiment 2 (just completed)
if [ -z "$exp1_checkpoint" ]; then
    exp1_checkpoint=$(ls -dt log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_* | head -1 | xargs -I {} find {} -name "best.pth" 2>/dev/null | head -1)
fi

# If still not found, use the existing best checkpoint from exp1 paper results
if [ -z "$exp1_checkpoint" ]; then
    exp1_checkpoint="log/psnet12/respointnet2_dp_fi_df_fc1_max_20250919163927/best.pth"
fi

if [ -f "$exp1_checkpoint" ]; then
    echo "Found Stage 1 checkpoint: $exp1_checkpoint"
    
    # Create a custom two-stage experiment script that uses the existing checkpoint
    cat > scripts/run_psnet12_two_stage_with_pretrained.sh << 'EOF'
#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
num_gpus=2
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage"
dataset_name='psnet12'
log_root="log/${dataset_name}"
master_port_train=12351  # Different port to avoid conflicts
master_port_eval=12352
patience=${1:-20} # Use the first argument for early stopping patience, default to 20
pretrained_checkpoint=${2:-""} # Second argument is the pretrained checkpoint path

# Two-stage specific parameters
stage1_epochs=80
stage2_epochs=20
stage2_lr=0.0001
use_cb_loss_stage1=false
use_cb_loss_stage2=true
freeze_encoder=true

echo "================================================="
echo "Starting Two-Stage training (with pretrained Stage 1)"
echo "Config: ${config_name}"
echo "Pretrained checkpoint: ${pretrained_checkpoint}"
echo "Stage 1 Epochs: ${stage1_epochs}, Stage 2 Epochs: ${stage2_epochs}"
echo "Stage 1 CB Loss: ${use_cb_loss_stage1}, Stage 2 CB Loss: ${use_cb_loss_stage2}"
echo "Freeze Encoder: ${freeze_encoder}"
echo "Using patience: ${patience}"
echo "================================================="

# Create experiment directory with timestamp
timestamp=$(date +%Y%m%d%H%M%S)
exp_dir="${log_root}/${config_name}_pretrained_${timestamp}"
mkdir -p "${exp_dir}/stage1"
mkdir -p "${exp_dir}/stage2"

# Copy the pretrained checkpoint as stage1_best.pth
if [ -f "${pretrained_checkpoint}" ]; then
    echo "Copying pretrained checkpoint to stage1 directory..."
    cp "${pretrained_checkpoint}" "${exp_dir}/stage1/stage1_best.pth"
    echo "Stage 1 checkpoint ready at: ${exp_dir}/stage1/stage1_best.pth"
else
    echo "Error: Pretrained checkpoint not found: ${pretrained_checkpoint}"
    exit 1
fi

# Run two-stage training (will skip Stage 1 since checkpoint exists)
time python -m torch.distributed.launch --master_port ${master_port_train} \
    --nproc_per_node ${num_gpus} \
    function/train_psnet_two_stage_dist.py \
    --dataset_name ${dataset_name} \
    --cfg cfgs/${dataset_name}/${config_name}.yaml \
    --log_dir "${exp_dir}" \
    --stage1_epochs ${stage1_epochs} \
    --stage2_epochs ${stage2_epochs} \
    --stage2_lr ${stage2_lr} \
    $([ "$use_cb_loss_stage1" = true ] && echo "--use_cb_loss_stage1") \
    $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss_stage2") \
    $([ "$freeze_encoder" = true ] && echo "--freeze_encoder") \
    --early_stopping_patience ${patience}

echo "Two-Stage training finished successfully."

# --- Find latest checkpoint directories ---
echo "================================================="
echo "Searching for the best checkpoints..."
echo "================================================="

# Check for Stage 1 and Stage 2 checkpoints
stage1_checkpoint="${exp_dir}/stage1/stage1_best.pth"
stage2_checkpoint="${exp_dir}/stage2/stage2_best.pth"

echo "Found experiment directory: ${exp_dir}"

# --- Evaluation of Stage 1 ---
if [ -f "${stage1_checkpoint}" ]; then
    echo "================================================="
    echo "Evaluating Stage 1 checkpoint..."
    echo "================================================="
    
    time python -m torch.distributed.launch --master_port ${master_port_eval} \
        --nproc_per_node 1 \
        function/evaluate_psnet_two_stage_dist.py \
        --dataset_name ${dataset_name} \
        --cfg cfgs/${dataset_name}/${config_name}.yaml \
        --load_path "${stage1_checkpoint}" \
        --stage stage1
    
    echo "Stage 1 evaluation finished."
else
    echo "Warning: Stage 1 checkpoint not found at ${stage1_checkpoint}"
fi

# --- Evaluation of Stage 2 ---
if [ -f "${stage2_checkpoint}" ]; then
    echo "================================================="
    echo "Evaluating Stage 2 (Final) checkpoint..."
    echo "================================================="
    
    time python -m torch.distributed.launch --master_port ${master_port_eval} \
        --nproc_per_node 1 \
        function/evaluate_psnet_two_stage_dist.py \
        --dataset_name ${dataset_name} \
        --cfg cfgs/${dataset_name}/${config_name}.yaml \
        --load_path "${stage2_checkpoint}" \
        --stage stage2 \
        $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss")
    
    echo "Stage 2 evaluation finished."
else
    echo "Warning: Stage 2 checkpoint not found at ${stage2_checkpoint}"
fi

echo "================================================="
echo "Two-Stage experiment complete."
echo "Results saved in: ${exp_dir}"
echo "================================================="
EOF

    # Make the script executable
    chmod +x scripts/run_psnet12_two_stage_with_pretrained.sh
    
    # Run the two-stage experiment with the pretrained checkpoint
    bash scripts/run_psnet12_two_stage_with_pretrained.sh 20 "$exp1_checkpoint"
else
    echo "ERROR: No suitable checkpoint found for Stage 1 pretraining!"
    echo "Please ensure either Experiment 1 or 2 completed successfully."
    exit 1
fi

echo "EXPERIMENT 3 COMPLETED"
echo ""

echo "=========================================="
echo "ALL 3 PAPER EXPERIMENTS COMPLETED"
echo "=========================================="
echo "Results summary:"
echo "1. Baseline ResPointNet++ (patience 50): Check latest log in log/psnet12/respointnet2_dp_fi_df_fc1_max_*"
echo "2. Strong CB Loss (patience 50): Check latest log in log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_*"
echo "3. Two-stage training: Check latest log in log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage_pretrained_*"
echo "=========================================="