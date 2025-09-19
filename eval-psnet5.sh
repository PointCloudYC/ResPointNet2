#!/bin/bash

num_gpus=2
config_name='respointnet2_dp_fi_df_fc1_max'
dataset_name='psnet5'
# check_point_file_name='/media/yinchao/code/ResPointNet2/log/psnet5/respointnet2_dp_fi_df_fc1_max_20250918160047/ckpt_epoch_80.pth'
check_point_file_name='/media/yinchao/code/ResPointNet2/log/psnet5/respointnet2_dp_fi_df_fc1_max_20250918160047/best.pth'


# eval psnet5
time python -m torch.distributed.launch --master_port 12345 \
--nproc_per_node 1 \
function/evaluate_psnet_dist.py \
--dataset_name ${dataset_name} \
--cfg cfgs/${dataset_name}/${config_name}.yaml \
--load_path ${check_point_file_name} # e.g., log/psnet/pointwisemlp_dp_fc1_1617578755/ckpt_epoch_190.pth