import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utlis import create_kernel_points, radius_gaussian, weight_variable
from pt_utils import MaskedQueryAndGroup

class ResPointNet2(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """A ResPointNet2 operator for local aggregation

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(ResPointNet2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nsample = nsample
        self.feature_type = config.respointnet2.feature_type
        self.feature_input_channels = {'dp_fj': 3 + in_channels,
                                       'fi_df': 2 * in_channels,
                                       'dp_fi_df': 3 + 2 * in_channels}
        self.feature_input_channels = self.feature_input_channels[self.feature_type]
        self.num_mlps = config.respointnet2.num_mlps
        self.reduction = config.respointnet2.reduction

        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True, normalize_xyz=True)

        self.mlps = nn.Sequential()
        if self.num_mlps == 1:
            self.mlps.add_module('conv0', nn.Sequential(
                nn.Conv2d(self.feature_input_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True)))
        else:
            mfdim = max(self.in_channels // 2, 9)
            self.mlps.add_module('conv0', nn.Sequential(
                nn.Conv2d(self.feature_input_channels, mfdim, kernel_size=1, bias=False),
                nn.BatchNorm2d(mfdim, momentum=config.bn_momentum),
                nn.ReLU(inplace=True)))
            for i in range(self.num_mlps - 2):
                self.mlps.add_module(f'conv{i + 1}', nn.Sequential(
                    nn.Conv2d(mfdim, mfdim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mfdim, momentum=config.bn_momentum),
                    nn.ReLU(inplace=True)))
            self.mlps.add_module(f'conv{self.num_mlps - 1}', nn.Sequential(
                nn.Conv2d(mfdim, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels, momentum=config.bn_momentum),
                nn.ReLU(inplace=True)))

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        neighborhood_features, relative_position, neighborhood_mask = self.grouper(query_xyz, support_xyz, query_mask,
                                                                                   support_mask, support_features)
        if self.feature_type == 'dp_fi_df':
            # B C N M
            center_features = torch.unsqueeze(neighborhood_features[..., 0], -1).repeat([1, 1, 1, self.nsample])
            relative_features = neighborhood_features - center_features
            local_input_features = torch.cat([relative_position, center_features, relative_features], 1)
            aggregation_features = self.mlps(local_input_features) # (B,C_out,N1,nsample)
        else:
            raise NotImplementedError(f'Feature Type {self.feature_type} not implemented in ResPointNet2')

        if self.reduction == 'max':
            out_features = F.max_pool2d(
                aggregation_features, kernel_size=[1, self.nsample] # (B,C_out, N1)
            )
            out_features = torch.squeeze(out_features, -1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None]) # (B,N1=15000,nsample) + (B,N1,1)
            # why feature_mask computes like this? each padded pt's nb mask will get 1, or 2?
            feature_mask = feature_mask[:, None, :, :] # (B,1,N1,nsample)
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
            neighborhood_num = feature_mask.sum(-1)
            out_features /= neighborhood_num
        elif self.reduction == 'sum':
            feature_mask = neighborhood_mask + (1 - query_mask[:, :, None])
            feature_mask = feature_mask[:, None, :, :]
            aggregation_features *= feature_mask
            out_features = aggregation_features.sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in ResPointNet2')
        return out_features

class LocalAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, radius, nsample, config):
        """LocalAggregation operators

        Args:
            in_channels: input channels.
            out_channels: output channels.
            radius: ball query radius
            nsample: neighborhood limit.
            config: config file
        """
        super(LocalAggregation, self).__init__()
        if config.local_aggregation_type == 'respointnet2':
            self.local_aggregation_operator = ResPointNet2(in_channels, out_channels, radius, nsample, config)
        else:
            raise NotImplementedError(f'LocalAggregation {config.local_aggregation_type} not implemented')

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, support_features):
        """
        Args:
            query_xyz: [B, N1, 3], query points.
            support_xyz: [B, N2, 3], support points.
            query_mask: [B, N1], mask for query points.
            support_mask: [B, N2], mask for support points.
            support_features: [B, C_in, N2], input features of support points.

        Returns:
           output features of query points: [B, C_out, 3]
        """
        return self.local_aggregation_operator(query_xyz, support_xyz, query_mask, support_mask, support_features)