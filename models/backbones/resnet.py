import os
import sys
from ..local_aggregation_operators import LocalAggregation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

import torch.nn as nn
from pt_utils import MaskedMaxPool


class MultiInputSequential(nn.Sequential):

    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input) # Nice use *array to pass in all
        return input


class Bottleneck(nn.Module):
    """local aggregation operator(LAO) with bottleneck(i.e., point cloud residual block version) and downsampling support; Note: LAO equivalent to Conv2D, with this module equivalent to ResNet w. bottleneck
    Args:
        nn ([type]): torch module
    """
    def __init__(self, in_channels, out_channels, bottleneck_ratio, radius, nsample, config,
                 downsample=False, sampleDl=None, npoint=None):
        """

        Args:
            in_channels (int): the number of input channel
            out_channels (int): the number of output channel
            bottleneck_ratio (int, optional): the bottleneck division ratio. Defaults to 2.
            radius (float): neighborhood search radius for local feature learning
            nsample (int): the number of neighboring points for current residual stage, e.g., 26
            config (dict): the config object
            downsample (bool, optional): whether to downsample. Defaults to False.
            sampleDl (float): voxel grid size
            npoint (int, optional): the number of downsampling points for current stage. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        if downsample:
            # process over a higher resolution (i.e., grid size(sampleDl) doubles)
            self.maxpool = MaskedMaxPool(npoint, radius, nsample, sampleDl) 

        # 1x1 conv, in_channels --> out_channels//bottleneck_ratio
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels // bottleneck_ratio, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels // bottleneck_ratio, momentum=config.bn_momentum),
                                   nn.ReLU(inplace=True))

        # LAO, out_channels// bottleneck_ratio--> out_channels// bottleneck_ratio
        self.local_aggregation = LocalAggregation(out_channels // bottleneck_ratio,
                                                  out_channels // bottleneck_ratio,
                                                  radius, nsample, config)

        # 1x1 conv, out_channels//bottleneck_ratio-->out_channels
        self.conv2 = nn.Sequential(nn.Conv1d(out_channels // bottleneck_ratio, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels, momentum=config.bn_momentum))

    def forward(self, xyz, mask, features):
        """

        Args:
            xyz ([type]): support xyz, (B,3,npoints[i]), e.g., npoints=[4096,1152,...]
            mask ([type]): mask
            features ([type]): support features, (B,C_in,npoint[i])

        Returns:
            query_xyz: queried xyz, if down-sampling, then return less points else return the original xyz
            query_mask: queried mask
            output([type]): learned features, if down-sample, (B,C_out,npoints[i+1]) else (B,C_out,npoints[i])
        """

        if self.downsample:
            # grid sub-sampling w. max pool strategy to gain sub features
            sub_xyz, sub_mask, sub_features = self.maxpool(xyz, mask, features)
            query_xyz = sub_xyz
            query_mask = sub_mask
            identity = sub_features
        else:
            query_xyz = xyz
            query_mask = mask
            identity = features

        output = self.conv1(features)
        output = self.local_aggregation(query_xyz, xyz, query_mask, mask, output)
        output = self.conv2(output)

        if self.in_channels != self.out_channels:
            identity = self.shortcut(identity)

        output += identity
        output = self.relu(output)

        return query_xyz, query_mask, output


class ResNet(nn.Module):
    def __init__(self, config, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        """Resnet Backbone

        Args:
            config: config file.
            input_features_dim: dimension for input feature.
            radius: the base ball query radius.
            sampleDl: the base grid length for sub-sampling.
            nsamples: neighborhood limits for each layer, a List of int.
            npoints: number of points after each sub-sampling, a list of int.
            width: the base channel num.
            depth: number of bottlenecks in one stage.
            bottleneck_ratio: bottleneck ratio.

        Returns:
            A dict of points, masks, features for each layer.
        """
        super(ResNet, self).__init__()

        self.input_features_dim = input_features_dim

        # 1x1 conv w/o local feature learning, (B, 72, N)->(B, width/2=72, N)  assume width=C=144
        self.conv1 = nn.Sequential(nn.Conv1d(input_features_dim, width // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(width // 2, momentum=config.bn_momentum),
                                   nn.ReLU(inplace=True))

        # local feature learning w/o bottleneck, (B, C/2, N)->(B, C/2, N) 
        self.la1 = LocalAggregation(width // 2, width // 2, radius, nsamples[0], config)

        # local feature learning w. bottleneck(RB), (B, C/2, N)->(B, C, N) 
        self.btnk1 = Bottleneck(width // 2, width, bottleneck_ratio, radius, nsamples[0], config)

        # layer1 = Strided RB + depth*RB
        self.layer1 = MultiInputSequential()
        sampleDl *= 2 # relevant to strided conv
        # Strided RB, (B, C, N)->(B, 2C, npoints[0]), similar are layer{2,3,4}s
        self.layer1.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[0], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[0]))
        radius *= 2
        width *= 2
        # RB, (B, 2C, npoints[0])->(B, 2C, npoints[0]), similar are layer(({2,3,4}
        for i in range(depth - 1):
            self.layer1.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[1], config))

        self.layer2 = MultiInputSequential()
        sampleDl *= 2
        # Strided RB, (B, 2C, npoints[0])->(B, 4C, npoints[1])
        self.layer2.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[1], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[1]))
        radius *= 2
        width *= 2
        # RB, (B, 4C, npoints[1])->(B, 4C, npoints[1])
        for i in range(depth - 1):
            self.layer2.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[2], config))

        self.layer3 = MultiInputSequential()
        sampleDl *= 2
        # Strided RB, (B, 4C, npoints[1])->(B, 8C, npoints[2])
        self.layer3.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[2], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[2]))
        radius *= 2
        width *= 2
        # RB, (B, 8C, npoints[2])->(B, 8C, npoints[2])
        for i in range(depth - 1):
            self.layer3.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[3], config))

        self.layer4 = MultiInputSequential()
        sampleDl *= 2
        # Strided RB, (B, 8C, npoints[2])->(B, 16C, npoints[3])
        self.layer4.add_module("strided_bottleneck",
                               Bottleneck(width, 2 * width, bottleneck_ratio, radius, nsamples[3], config,
                                          downsample=True, sampleDl=sampleDl, npoint=npoints[3]))
        radius *= 2
        width *= 2
        # RB, (B, 16C, npoints[3])->(B, 16C, npoints[3])
        for i in range(depth - 1):
            self.layer4.add_module(f"bottlneck{i}",
                                   Bottleneck(width, width, bottleneck_ratio, radius, nsamples[4], config))

    def forward(self, xyz, mask, features, end_points=None):
        """
        Args:
            xyz: (B, N, 3), point coordinates
            mask: (B, N), 0/1 mask to distinguish padding points1
            features: (B, 3, input_features_dim), input points features.
            end_points: a dict

        Returns:
            end_points: a dict contains all outputs
        """
        if not end_points: end_points = {}

        # res1 (no down-sampling but increase channels to higher dims e.g., 72)
        features = self.conv1(features) # 1x1 conv, B 72 N
        features = self.la1(xyz, xyz, mask, mask, features)# aggregate info from nbhood using local aggregator w/o bottleneck, (B, 72, N)
        xyz, mask, features = self.btnk1(xyz, mask, features)# aggregate info from nbhood using local aggregator w. bottleneck, (B, 72, N)
        end_points['res1_xyz'] = xyz
        end_points['res1_mask'] = mask
        end_points['res1_features'] = features

        # res2 (downsample and simultaneously increase double channels), similar are res{3,4,5}
        xyz, mask, features = self.layer1(xyz, mask, features)
        end_points['res2_xyz'] = xyz
        end_points['res2_mask'] = mask
        end_points['res2_features'] = features

        # res3
        xyz, mask, features = self.layer2(xyz, mask, features)
        end_points['res3_xyz'] = xyz
        end_points['res3_mask'] = mask
        end_points['res3_features'] = features

        # res4
        xyz, mask, features = self.layer3(xyz, mask, features)
        end_points['res4_xyz'] = xyz
        end_points['res4_mask'] = mask
        end_points['res4_features'] = features

        # res5
        xyz, mask, features = self.layer4(xyz, mask, features)
        end_points['res5_xyz'] = xyz
        end_points['res5_mask'] = mask
        end_points['res5_features'] = features

        return end_points
