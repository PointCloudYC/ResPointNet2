# ResPointNet2 (Automated semantic segmentation of industrial point clouds using ResPointNet++)

by Chao Yin, Boyu Wang, [Vincent J.L.Gan](https://scholar.google.com.hk/citations?user=lBQYX9sAAAAJ&hl=zh-TW), [Mingzhu Wang*](https://www.lboro.ac.uk/departments/abce/staff/mingzhu-wang/), [Jack C.P.Cheng*](https://facultyprofiles.hkust.edu.hk/profiles.php?profile=jack-chin-pang-cheng-cejcheng)

## Abstract

Currently, as-built building information modeling (BIM) models from point clouds show great potential in managing building information. The automatic creation of as-built BIM models from point clouds is important yet challenging due to the inefficiency of semantic segmentation. To overcome this challenge, this paper proposes a novel deep learning-based approach, ResPointNet++, by integrating deep residual learning with conventional PointNet++ network. To unleash the power of deep learning methods, this study firstly builds an expert-labeled high-quality industrial LiDAR dataset containing 80 million data points collected from four different industrial scenes covering nearly 4000 m2. Our dataset consists of five typical semantic categories of plumbing and structural components (i.e., pipes, pumps, tanks, I-shape and rectangular beams). Second, we introduce two effective neural modules including local aggregation operator and residual bottleneck modules to learn complex local structures from neighborhood regions and build up deeper point cloud networks with residual settings. Based on these two neural modules, we construct our proposed network, ResPointNet++, with a U-Net style encoder-decoder structure. To validate the proposed method, comprehensive experiments are conducted to compare the robustness and efficiency of our ResPointNet++ with two representative baseline methods (PointNet and PointNet++) on our benchmark dataset. The experimental results demonstrate that ResPointNet++ outperforms two baselines with a remarkable overall segmentation accuracy of 94% and mIoU of 87%, which is 23% and 42% higher than that of conventional PointNet++, respectively. Finally, ablation studies are performed to evaluate the influence of design choices of the local aggregation operator module including input feature type and aggregation function type. This study contributes to automated 3D scene interpretation of industrial point clouds as well as the as-built BIM creation for industrial components such as pipes and beams.

## Requirements

To install requirements:

```setup
#!/bin/bash
ENV_NAME='respointnet2'
conda create –n $ENV_NAME python=3.6.10 -y
source activate $ENV_NAME
conda install -c anaconda pillow=6.2 -y
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
conda install -c conda-forge opencv -y
pip3 install termcolor tensorboard h5py easydict
```

Note that: the latest codes are tested on two Ubuntu settings: 
- Ubuntu 18.04, Nvidia 3090 (24Gb), CUDA 11.3, PyTorch 1.4 and Python 3.6
- Ubuntu 18.04, Nvidia 1080 (8Gb), CUDA 10.1, PyTorch 1.4 and Python 3.6

### Compile custom CUDA operators

```bash
sh init.sh
```

## PSNet5 dataset

PSNet5 dataset contains 80 million data points collected from four different industrial scenes covering nearly 4000 m^2. Five typical semantic categories of plumbing and structural classes are annotaed, including pipe, pump, tank, I-shape beam (ibeam) and rectangular beam(rbeam).

Click [here (onedrive link)](https://hkustconnect-my.sharepoint.com/:f:/g/personal/cyinac_connect_ust_hk/EgRPTDHPwkJNgv_PPhi9iioBqH6f6cEelR00TGRSfKzAVA?e=Vx2Qnl) to download our dataset, which is about 0.5Gb large and costs about 2.3 Gb after the unzipping.

- download the dataset and unzip the file to `root/data/PSNet`
- The file structure should look like:

```
<root>
├── cfgs
│   └── psnet5
├── data
│   └── PSNet
│       └── PSNet5
│           ├── Area_1
│           ├── Area_2
│           └── ...
├── init.sh
├── datasets
├── function
├── models
├── ops
└── utils
```

## Training

To train the model(s) in the paper, run this command or check the `train-psnet5.sh` for details.

```train
time python -m torch.distributed.launch --master_port 12346 \
--nproc_per_node ${num_gpus} \
function/train_psnet_dist.py \
--dataset_name ${dataset_name} \
--cfg cfgs/${dataset_name}/${config_name}.yaml
```


## Evaluation

To evaluate the model on PSNet5, run this command or check the `train-psnet5.sh`

```eval
time python -m torch.distributed.launch --master_port 12346 \
--nproc_per_node ${num_gpus} \
function/train_psnet_dist.py \
--dataset_name ${dataset_name} \
--cfg cfgs/${dataset_name}/${config_name}.yaml
```

`bash train-psnet5.sh`

## Pre-trained Models

You can download pretrained models here:

- [TOADD](https://drive.google.com/mymodel.pth) trained on the PSNet5

## Primary results

TODOs


## Acknowledgements

Our codes borrowed a lot from [CloserLook3D](https://github.com/zeliu98/CloserLook3D), [KPConv-pytorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch), [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Our other relevant works

- [SE-PseudoGrid: Automated Classification of Piping Components from 3D LiDAR Point Clouds](https://github.com/PointCloudYC/se-pseudogrid)

## License

Our code is released under MIT License (see LICENSE file for details).

## Citation

If you find our work useful in your research, please consider citing:

```
@article{respointnet2,
    Author = {C. Yin, B. Wang, V. J. L. Gan, M. Wang*, and J. C. P. Cheng*},
    Title = {Automated semantic segmentation of industrial point clouds using ResPointNet++},
    Journal = {Automation in Construction},
    Year = {2021}
    doi = {https://doi.org/10.1016/j.autcon.2021.103874}
   }
```