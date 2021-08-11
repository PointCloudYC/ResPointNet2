# ResPointNet2 (Automated semantic segmentation of industrial point clouds using ResPointNet++)

by Chao Yin, Boyu Wang, Vincent J.L.Gan, Mingzhu Wang, Jack C.P.Cheng

## Abstract

Currently, as-built building information modeling (BIM) models from point clouds show great potential in managing building information. The automatic creation of as-built BIM models from point clouds is important yet challenging due to the inefficiency of semantic segmentation. To overcome this challenge, this paper proposes a novel deep learning-based approach, ResPointNet++, by integrating deep residual learning with conventional PointNet++ network. To unleash the power of deep learning methods, this study firstly builds an expert-labeled high-quality industrial LiDAR dataset containing 80 million data points collected from four different industrial scenes covering nearly 4000 m2. Our dataset consists of five typical semantic categories of plumbing and structural components (i.e., pipes, pumps, tanks, I-shape and rectangular beams). Second, we introduce two effective neural modules including local aggregation operator and residual bottleneck modules to learn complex local structures from neighborhood regions and build up deeper point cloud networks with residual settings. Based on these two neural modules, we construct our proposed network, ResPointNet++, with a U-Net style encoder-decoder structure. To validate the proposed method, comprehensive experiments are conducted to compare the robustness and efficiency of our ResPointNet++ with two representative baseline methods (PointNet and PointNet++) on our benchmark dataset. The experimental results demonstrate that ResPointNet++ outperforms two baselines with a remarkable overall segmentation accuracy of 94% and mIoU of 87%, which is 23% and 42% higher than that of conventional PointNet++, respectively. Finally, ablation studies are performed to evaluate the influence of design choices of the local aggregation operator module including input feature type and aggregation function type. This study contributes to automated 3D scene interpretation of industrial point clouds as well as the as-built BIM creation for industrial components such as pipes and beams.

## Citation
C. Yin, B. Wang, V. J. L. Gan, M. Wang, and J. C. P. Cheng, “Automated semantic segmentation of industrial point clouds using ResPointNet++,” Automation in Construction, vol. 130, p. 103874, 2021, doi: https://doi.org/10.1016/j.autcon.2021.103874.

## Our dataset (industrial real-world point cloud dataset)

This is an expert-labeled high-quality industrial LiDAR dataset containing 80 million data points collected from four different industrial scenes covering nearly 4000 m^2. Our dataset consists of five typical semantic categories of plumbing and structural components (i.e., pipes, pumps, tanks, I-shape and rectangular beams).

To be released soon.

## Primary results

Stay tuned!


## Acknowledgements

Our codes borrowed a lot from [CloserLook3D](https://github.com/zeliu98/CloserLook3D), [KPConv-pytorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch), [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch).

## License

Our code is released under MIT License (see LICENSE file for details).
