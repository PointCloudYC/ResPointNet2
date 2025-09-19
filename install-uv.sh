#!/bin/bash

# create uv environment
# uv venv -p 3.10
# source .venv/bin/activate

# install dependencies
# 250915, for newer versions of nvidia drivers(e.g., 550) with CUDA12.1+, 1) use pytorch 1.13.1 with CUDA toolkit 11.7; 2) make sure your system has installed the system-wide CUDA toolkit 11.7. details on how to install two CUDA toolkit versions on your system, see https://medium.com/@yushantripleseven/managing-multiple-cuda-cudnn-installations-ba9cdc5e2654#31dc and yc/faq.md
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install scikit-learn IPython scikit-image matplotlib pillow tqdm opencv-python setuptools ipykernel
uv pip install termcolor tensorboard h5py easydict

# install custom operators
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