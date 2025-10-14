## temp todos
**FSL methods**
- [x] allow to run PosPool, PseudoGrid(KPConv) into the ResPointNet++ repo: 1) config, 2)dataset, 3)model, 4)others
  - [x] Added PosPool and PseudoGrid classes to models/local_aggregation_operators.py
  - [x] Created config files: pospool_xyz_avg_a6000.yaml, pospool_sin_cos_avg_a6000.yaml, pseudo_grid_a6000.yaml
  - [x] Created training scripts: run_psnet12_pospool_a6000.sh, run_psnet12_pseudogrid_a6000.sh
  - [x] Updated utils/config.py with default config structures
  - [x] Created documentation in docs/NEW_MODELS.md

# Industrial3D Experiment TODOs (Deadline: 2025-10-30)

This list tracks the experiments planned for the Industrial3D benchmark.

### Phase 1: Setup & Prerequisites
- [x] Create isolated `conda`/`docker` environments for all models.
- [x] Finalize data preparation and splitting scripts.
- [x] Implement a unified evaluation script for all metrics.

### Phase 2: Model Benchmarking

**Fully-Supervised Learning (FSL)**
- [x] **ResPointNet++:** Adapt, train, and evaluate.
- [x] **PseudoGrid (KPConv):** Adapt complete. Ready to train and evaluate.
- [x] **PosPool:** Adapt complete. Ready to train and evaluate.
- [ ] **RandLA-Net:** Adapt, train, and evaluate.

**Weakly-Supervised Learning (WSL)**
- [ ] **SQN:** Adapt, train (1%, 5%, 10% labels), and evaluate.
- [ ] **CPCM:** Adapt, train (1%, 5%, 10% labels), and evaluate. *(Stretch Goal)*

**Self-Supervised Learning (SSL)**
- [ ] **OcOc:** Pre-train, fine-tune (1%, 5%, 10% labels), and evaluate.
- [ ] **GrowSP/LogoSP or PointCept:** Choose one, adapt, pre-train, fine-tune, and evaluate. *(User to decide which one)*

**Foundation Models (FM)**
- [ ] **Point-SAM:**
    - [ ] Evaluate in zero-shot setting.
    - [ ] Fine-tune and evaluate in few-shot setting (1%, 5%, 10% labels).

### Phase 3: Analysis & Reporting
- [ ] **Domain Challenge: Robustness:** Analyze model performance on cluttered/occluded data.
- [ ] **Domain Challenge: Ambiguity:** Analyze confusion matrices and multi-modal input impact.
- [ ] **Domain Challenge: Sparse Supervision:** Compare performance of WSL, SSL, and FMs with few labels.
- [ ] Consolidate all results into tables for the paper.
- [ ] Generate qualitative result visualizations.
- [ ] Update `main.tex` with final results and findings.



## problems
### Refactor PSNet12 dataset loading for cross-platform compatibility

- reason: Refactor PSNet12 dataset loading for cross-platform compatibility; If processed files are reused, the KDTrees will be different across platforms, leading to errors.
- solution: Rebuild KDTrees on load in datasets/PSNet12.py


### why ResPointNet++ is so slow? 

- reason: the system is under malware attack, the system is very slow. Note: also from the program side, the ResPointNet++ is very slow due to reusing the previous KDTrees from the previous server, see Refactor PSNet12 dataset loading for cross-platform compatibility.
- problem
	- Continuous high CPU load (> 100 ×) from repeating `-bash` processes owned by `yc`.
	- Killing the process or removing tmux/zsh sessions did not stop it.
	- New PIDs appeared every few seconds and eventually saturated all cores.
	- System load > 100, memory heavily used, making the server unusable for others.
- root cause: A **malicious cron job** under user `yc` was launching a fake shell wrapper every minute.
	- That directory mimicked a legitimate X11 path but contained a script (`upd`) that spawned many background `bash` processes — essentially a fork bomb / persistence implant.
	- The same binary name `zsh` in `~/local/bin/zsh` was also a wrapper calling `bash`, keeping the loop alive.
	- The malware maintained persistence through the user’s **crontab**, surviving reboot and manual kills.
- sol.:
```bash
crontab -r  # or crontab -e and delete suspicious line
rm -rf /var/tmp/.ICE-Unix/-bash
pkill -9 -f '/var/tmp/.ICE-Unix/-bash'
pkill -9 -u yc

Remove the fake `~/local/bin/zsh`
Install genuine `/usr/bin/zsh` or rebuild from package/source.

loginctl terminate-user yc

`ps -u yc`, `top` # check status
```
- future todos, see: https://i.askurl.cn/8aeT0vp8

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
- 2) iJIT_NotifyEvent symbol error indicates a missing Intel VTune dependency. Use the following command to compile the ops(use CUDA 11.7 to compile the ops rather the default CUDA (which is 12.4)):
```bash
cd ops/pt_custom_ops
export CUDA_HOME=/usr/local/cuda-11.7
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
python setup.py install --user
```
refs: https://i.askurl.cn/c3HVwVyU

### numpy version is too high

- Installed numpy 1.26.4 and six 1.17.0 in the conda environment


### conda installing is very slow

- use tsinghua source and mamaba to install packages
```bash
写入 ~/.condarc 为清华镜像, vi ~/.condarc, 写入以下内容:
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
show_channel_urls: true
channel_priority: flexible

安装 mamba：conda install -n base -c conda-forge mamba -y
之后用 mamba 安装：mamba install matplotlib -y
```