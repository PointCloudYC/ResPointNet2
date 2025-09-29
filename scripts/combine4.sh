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

# --- Configuration ---
num_gpus=2
config_name="respointnet2_dp_fi_df_fc1_max_cb_loss_two_stage"
dataset_name='psnet12'
log_root="log/${dataset_name}"
master_port_train=12351  # Different port to avoid conflicts
master_port_eval=12352
patience=${1:-20} # Use the first argument for early stopping patience, default to 20
# best checkpoint from normal CE loss

# two pretrained checkpoints: training with normal CE loss and strong CB loss
pretrained_checkpoints=("log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth" "log/psnet12/respointnet2_dp_fi_df_fc1_max_20250924205054/best.pth")
# best checkpoint from strong CB loss
# pretrained_checkpoint="log/psnet12/respointnet2_dp_fi_df_fc1_max_cb_loss_strong_20250926015005/best.pth"
# pretrained_checkpoint=${2:-""} # Second argument is the pretrained checkpoint path

# Two-stage specific parameters
stage1_epochs=80
stage2_epochs=20
stage2_lrs=(0.0005 0.001 0.005)
use_cb_loss_stage1=false
use_cb_loss_stage2=true
freeze_encoder=true

# Run two-stage training (will skip Stage 1 since checkpoint exists)
for pretrained_checkpoint in "${pretrained_checkpoints[@]}"; do
    for stage2_lr in "${stage2_lrs[@]}"; do
        echo "================================================="
        echo "Starting Two-Stage training (with pretrained Stage 1 from strong CB loss's best checkpoint)"
        echo "Pretrained checkpoint: ${pretrained_checkpoint}"
        echo "Stage2 LR: ${stage2_lr}"
        echo "Config: ${config_name}"
        echo "Stage 1 Epochs: ${stage1_epochs}, Stage 2 Epochs: ${stage2_epochs}"
        echo "Stage 1 CB Loss: ${use_cb_loss_stage1}, Stage 2 CB Loss: ${use_cb_loss_stage2}"
        echo "Freeze Encoder: ${freeze_encoder}"
        echo "Using patience: ${patience}"
        echo "================================================="

        # Create experiment directory with timestamp
        timestamp=$(date +%Y%m%d%H%M%S)
        exp_dir="${log_root}/${config_name}_pretrained_${timestamp}_${stage2_lr}"
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
    done
done

echo "Two-Stage training all experiments finished successfully."


# # --- Find latest checkpoint directories ---
# echo "================================================="
# echo "Searching for the best checkpoints..."
# echo "================================================="
# # Check for Stage 1 and Stage 2 checkpoints
# stage1_checkpoint="${exp_dir}/stage1/stage1_best.pth"
# stage2_checkpoint="${exp_dir}/stage2/stage2_best.pth"

# echo "Found experiment directory: ${exp_dir}"

# # --- Evaluation of Stage 1 ---
# if [ -f "${stage1_checkpoint}" ]; then
#     echo "================================================="
#     echo "Evaluating Stage 1 checkpoint..."
#     echo "================================================="
    
#     time python -m torch.distributed.launch --master_port ${master_port_eval} \
#         --nproc_per_node 1 \
#         function/evaluate_psnet_two_stage_dist.py \
#         --dataset_name ${dataset_name} \
#         --cfg cfgs/${dataset_name}/${config_name}.yaml \
#         --load_path "${stage1_checkpoint}" \
#         --stage stage1
    
#     echo "Stage 1 evaluation finished."
# else
#     echo "Warning: Stage 1 checkpoint not found at ${stage1_checkpoint}"
# fi

# # --- Evaluation of Stage 2 ---
# if [ -f "${stage2_checkpoint}" ]; then
#     echo "================================================="
#     echo "Evaluating Stage 2 (Final) checkpoint..."
#     echo "================================================="
    
#     time python -m torch.distributed.launch --master_port ${master_port_eval} \
#         --nproc_per_node 1 \
#         function/evaluate_psnet_two_stage_dist.py \
#         --dataset_name ${dataset_name} \
#         --cfg cfgs/${dataset_name}/${config_name}.yaml \
#         --load_path "${stage2_checkpoint}" \
#         --stage stage2 \
#         $([ "$use_cb_loss_stage2" = true ] && echo "--use_cb_loss_stage2")
    
#     echo "Stage 2 evaluation finished."
# else
#     echo "Warning: Stage 2 checkpoint not found at ${stage2_checkpoint}"
# fi

# echo "================================================="
# echo "Two-Stage experiment complete."
# echo "Results saved in: ${exp_dir}"
# echo "================================================="

