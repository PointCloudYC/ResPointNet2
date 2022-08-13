#!/bin/bash

num_gpus=2
dataset_name='psnet12'
config_name="respointnet2_dp_fi_df_fc1_max"

# train psnet5
time python -m torch.distributed.launch --master_port 12346 \
--nproc_per_node ${num_gpus} \
function/train_psnet_dist.py \
--dataset_name ${dataset_name} \
--cfg cfgs/${dataset_name}/${config_name}.yaml
# [--log_dir <log directory>]


# eval psnet5
# time python -m torch.distributed.launch --master_port 123456 \
# --nproc_per_node 1 \
# function/evaluate_psnet_dist.py \
# --dataset_name ${dataset_name} \
# --cfg cfgs/${dataset_name}/${config_name}.yaml
# --load_path {check_point_file_name}, # e.g., log/psnet/pointwisemlp_dp_fc1_1617578755/ckpt_epoch_190.pth
