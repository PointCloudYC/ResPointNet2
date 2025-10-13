#!/bin/bash

ENV_NAME='respointnet2'

conda create –n $ENV_NAME python=3.10 -y
source activate $ENV_NAME

# 250915, for newer versions of nvidia drivers(e.g., 550) with CUDA12.1+, 1) use pytorch 1.13.1 with CUDA toolkit 11.7; 2) make sure your system has installed the system-wide CUDA toolkit 11.7. details on how to install two CUDA toolkit versions on your system, see https://medium.com/@yushantripleseven/managing-multiple-cuda-cudnn-installations-ba9cdc5e2654#31dc and yc/faq.md
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

conda install -c conda-forge opencv -y
conda install -c anaconda pillow -y
conda install -c conda-forge  termcolor tensorboard h5py easydict scikit-learn -y
# pip3 install termcolor tensorboard h5py easydict scikit-learn
# install open3d and plyfile for visualization
conda install -c open3d-admin open3d -y
conda install -c conda-forge plyfile -y


# install custom operators

# assume CUDA 11.7 is installed and set as the default CUDA on the system
# if not, please install CUDA 11.7 and set it as the default CUDA on the system
# export CUDA_HOME=/usr/local/cuda-11.7
# export CUDA_PATH=$CUDA_HOME
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
cd ops/cpp_wrappers
sh compile_wrappers.sh
cd ../pt_custom_ops
python setup.py install --user
cd ../..

# pre-processing all datasets, you can modify it according to your needs.
mkdir -p data/PSNet
ln -s /mnt/nas/point-cloud-datasets/PSNet5 PSNet5
python datasets/PSNet5.py
python datasets/PSNet12.py
