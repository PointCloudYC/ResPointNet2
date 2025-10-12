## Benchmarking Industrial3D on full spectrum of DL paradigms(Deadline: 2025-10-25)

- FSL
   - ResPointNet++
   - PosPool
   - PseudoGrid(KPConv)
   - RandLA-Net
   - PT v3
   - KPConvX
- WSL
   - SQN
   - CPCM
- USL
   - OcOc
   - Jaw3D
- FM
   - Point-Bert
   - Point-SAM
   - ?


## problems

### adjust settings for the new server (A6000)

- A6000 optimizations in cfgs/psnet12/respointnet2_dp_fi_df_fc1_max_a6000.yaml:
   - Increased batch size: 6 → 16 (2.67x) to utilize 48GB VRAM
   - Increased workers: 4 → 8 for better data loading throughput
   - Scaled learning rate: 0.01 → 0.0267 following linear scaling rule
   - Added warmup epoch: 5 for large batch stability
   - Enabled mixed precision (amp_opt_level: 'O1') for ~2x speedup
   - Note: num_points remains at 15000 (commented out increase to 20000)
- the experiments are run in the scripts/run_psne12_exp_a6000.sh

### compiling ops failed

- 1) make sure the setuptools version < 60.0.0; `conda run -n torch pip install "setuptools<60.0"`, then the cpp_sampler is compiled successfully
- 2) iJIT_NotifyEvent symbol error indicates a missing Intel VTune dependency. This is a known issue with certain PyTorch builds; `conda install intel-vtune-python-sdk -c intel`