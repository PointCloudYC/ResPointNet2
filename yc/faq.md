## can not compile ops@250915

- prob.: tittle; reporting 1)nvcc CUDA version was not compatible with pytorch; 2) still errors;
- sol.: make sure your conda environment has a built-in nvcc environment and install a system-wide CUDA toolkit (Note: I already have 12.1 installed, so I need manage cuda toolkit versions)

```bash
# install system-wide CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
# only install the toolkit as u already have a newer nvidia driver installed and it also supports cuda 11.7
sudo sh cuda_11.7.0_515.43.04_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.7
# verify the installation
/usr/local/cuda-11.7/bin/nvcc --version

# conda install pytorch and cuda toolkit
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# manage cuda toolkit versions 
# way 1 for execute 1 command
CUDA_HOME=/usr/local/cuda-11.7 python setup.py install
# way 2 for execute in a command prompt
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
nvcc --version
# way 3 make the default link to cuda 11.7
# sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.7 /usr/local/cuda
nvcc --version

# compile ops
bash init.sh
```

- refs
  - gemini chat: https://g.co/gemini/share/18284f7e5948
  - [Managing Multiple CUDA + cuDNN Installations | by YushanT7 | Medium](https://medium.com/@yushantripleseven/managing-multiple-cuda-cudnn-installations-ba9cdc5e2654#31dc), main useful guide, using sh files to install cuda toolkit.
  - [Managing multiple CUDA versions using environment modules in Ubuntu](https://gist.github.com/garg-aayush/156ec6ddda3d62e2c0ddad00b7e66956), good guide, using deb files to install cuda toolkit.
  - [pointnet2\_ops install problem · Issue #174 · erikwijmans/Pointnet2\_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174)
  - [CUDA Installation Guide for Linux — Installation Guide for Linux 13.0 documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-cuda-using-conda)
  - [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)


  ## miou is alsways zero for the baseline training@250924

  - prob.: tittle; reporting 1)miou is alsways zero for the baseline training;
  - sol.:  dist.barrier() pytorch 1.13.1 has no timeout parameter, so we need to use dist.barrier() to sync the processes.

  ## can not resume training for two-stage training@250924

  - prob.: tittle; reporting 1)can not resume training;
  - sol.: make sure the log_dir is the same as the original training;

  ## implementation the tail-class aware sampling@250924

  - prob.: tittle; 
  - sol.:  refer to the core.md and cb_sampler.py;


  ## decoupled training vs. curriculum learning@250924

They both stem from the idea that the standard, random-order (i.i.d.) training paradigm is not always optimal. However, they structure the learning process in different ways to achieve different goals.

---
The Shared Intuition: Structured Learning 🤔
Both **Decoupled Training** and **Curriculum Learning (CL)** are built on the idea of making the learning problem easier for the model by presenting information in a structured, non-random order. They both move from a "simpler" phase to a "harder" phase.
* In **Curriculum Learning**, "simple" means easy-to-classify examples.
* In **Decoupled Training**, "simple" means learning general features from all data without worrying about class imbalance.

---

Key Differences: What vs. How

The primary difference lies in *what* is being manipulated and for *what purpose*. A simple table makes this clear:

| Feature                   | **Curriculum Learning (CL)** | **Decoupled Training (for LTL)** |
| ------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------- |
| **Primary Goal** | Guide optimization to a better minimum and speed up convergence. | **Debias the classifier** that is skewed by the long-tailed data.  |
| **What is Manipulated?** | The **order and difficulty of data samples**.                | The **training stages and model components** (e.g., freezing the encoder). |
| **Data Progression** | Starts with "easy" samples, then gradually introduces "harder" ones. | Starts with the **full, imbalanced dataset**, then switches to a **class-balanced subset**. |
| **Model & Objective** | The model architecture and loss function typically remain **fixed** throughout. | The model's state **changes** (layers are frozen) and the data distribution for training changes. |

---

### ## Relationship: Are They the Same or a Subset? 🍎🍊

No, they are not the same, and decoupled training is **not considered a subset of curriculum learning**.

Think of them as different teaching philosophies:
* **Curriculum Learning** is like teaching a child math by starting with `1+1=2` before moving to algebra. You order the topics from easy to hard, but the student (the model) is always the same.
* **Decoupled Training** is like a university student's learning process. First, they attend all lectures for a semester to gain broad foundational knowledge (**Stage 1: Representation**). Then, they do targeted, intense practice on past exam questions, focusing on the tricky topics they know will appear (**Stage 2: Classifier Re-training**). The *phase* and *goal* of learning have changed.

**Conclusion:** Decoupled training is best described as a **multi-stage training strategy** specifically designed to solve the problem of representation-classifier conflict in long-tailed datasets. While it shares the spirit of structured learning with curriculum learning, its mechanics and primary motivation are distinct.