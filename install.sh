#!/bin/bash

ENV_NAME='respointnet2'

conda create –n $ENV_NAME python=3.6.10 -y
source activate $ENV_NAME

conda install -c anaconda pillow=6.2 -y
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch -y
conda install -c conda-forge opencv -y
pip3 install termcolor tensorboard h5py easydict