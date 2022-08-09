""" 
based on paper titled replicate semantic query network(https://arxiv.org/abs/2104.04891)
author: Chao YIN, cyinac@connect.ust.hk
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pt_utils import TrilinearInterpolateSQN, MaskedUpsample



class SqnHeadResNet(nn.Module):
    def __init__(self, num_classes,weak_supervision_ratio=0.1, num_neighbors=3):
        """A SQN head for ResNet backbone.

        Args:
            num_classes: class num.
            nsamples: neighborhood limits for each layer, a List of int.

        Returns:
            logits: (B, num_classes, N)
        """
        super(SqnHeadResNet, self).__init__()
        self.num_classes = num_classes
        self.weak_supervision_ratio = weak_supervision_ratio
        self.num_neighbors = num_neighbors

        self.query1= TrilinearInterpolateSQN(num_neighbors=3,ret_unknown_features=False)
        self.query2= TrilinearInterpolateSQN(num_neighbors=3,ret_unknown_features=False)
        self.query3= TrilinearInterpolateSQN(num_neighbors=3,ret_unknown_features=False)
        self.query4= TrilinearInterpolateSQN(num_neighbors=3,ret_unknown_features=False)
        self.query5= TrilinearInterpolateSQN(num_neighbors=3,ret_unknown_features=False)

        # MLP(4464=T, T/4, T/16, T/64, num_classes)
        width = 4464
        # remove batch norm since batch size might 1 (due to the weakly point might 1 in a original batch)
        self.head = nn.Sequential(nn.Conv1d(width, width // 4, kernel_size=1, bias=False),
                                  # nn.BatchNorm1d(width // 4),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 4, width // 16, kernel_size=1, bias=False),
                                  # nn.BatchNorm1d(width // 16),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 16, width // 64, kernel_size=1, bias=False),
                                  # nn.BatchNorm1d(width // 64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 64, num_classes, kernel_size=1, bias=True))

    def forward(self, weakly_points, end_points, batch_inds):
        """[summary]
        Args:
            weakly_points ([type]): the query point, (n,3)
            end_points ([type]): a dict contains all residual stages, each stage contains {res1_xyz,res1_mask, res1_features}
            batch_inds: (B,) tensor of the batch indices to denote which batch for the unknown, values are 0 to B-1

        Returns:
            logits: predicted scores, (n',num_classes)
        """
        feature1 = self.query1(weakly_points, end_points['res1_xyz'],batch_inds, unknow_feats=None, known_feats=end_points['res1_features']) # (n,C1,1)
        feature2 = self.query2(weakly_points, end_points['res2_xyz'],batch_inds, unknow_feats=None, known_feats=end_points['res2_features'])
        feature3 = self.query3(weakly_points, end_points['res3_xyz'],batch_inds, unknow_feats=None, known_feats=end_points['res3_features'])
        feature4 = self.query4(weakly_points, end_points['res4_xyz'],batch_inds, unknow_feats=None, known_feats=end_points['res4_features'])
        feature5 = self.query5(weakly_points, end_points['res5_xyz'],batch_inds, unknow_feats=None, known_feats=end_points['res5_features'])
        # features = self.up0(end_points['res4_xyz'], end_points['res5_xyz'],end_points['res4_mask'], end_points['res5_mask'], end_points['res5_features'])


        # concat all features, (n, 4464, 1)
        features_combined = torch.cat([feature1, feature2, feature3, feature4, feature5],dim=1)

        # (n, C/4, 1)-> (n, C/16, 1)->(n, C/64, 1)-->(n, num_classes, 1)
        logits = self.head(features_combined)

        return logits.squeeze(dim=-1) # (n, num_classes)