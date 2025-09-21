#!/bin/bash

BATCH_SIZES=(6 4 2)
CFGS=("respointnet2_dp_fi_df_fc1_max")

# train psnet5
for batch_size in "${BATCH_SIZES[@]}"; do
	for cfg in "${CFGS[@]}"; do
		echo "config file: ${cfg}"
		echo "batch size: ${batch_size}"

		time python -m torch.distributed.launch \
		--master_port 12346 \
		--nproc_per_node 1 \
		function/train_psnet5_dist.py \
		--cfg cfgs/psnet5/${cfg}.yaml \
		--batch_size ${batch_size} \
		--save_freq 10 \
		--val_freq 10 

		echo "config file: ${cfg}"
		echo "batch size: ${batch_size}"
	done
done
