import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'ops', 'pt_custom_ops'))

from pt_utils import MaskedUpsample


class SceneSegHeadResNet(nn.Module):
    def __init__(self, num_classes, width, base_radius, nsamples):
        """A scene segmentation head for ResNet backbone.

        Args:
            num_classes: class num.
            width: the base channel num.
            base_radius: the base ball query radius.
            nsamples: neighborhood limits for each layer, a List of int.

        Returns:
            logits: (B, num_classes, N)
        """
        super(SceneSegHeadResNet, self).__init__()
        self.num_classes = num_classes
        self.base_radius = base_radius
        self.nsamples = nsamples
        # 16r->8r, other up-sampling layers are similar
        self.up0 = MaskedUpsample(radius=8 * base_radius, nsample=nsamples[3], mode='nearest')
        self.up1 = MaskedUpsample(radius=4 * base_radius, nsample=nsamples[2], mode='nearest')
        self.up2 = MaskedUpsample(radius=2 * base_radius, nsample=nsamples[1], mode='nearest')
        self.up3 = MaskedUpsample(radius=base_radius, nsample=nsamples[0], mode='nearest')

        self.up_conv0 = nn.Sequential(nn.Conv1d(24 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(4 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv1 = nn.Sequential(nn.Conv1d(8 * width, 2 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(2 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv1d(4 * width, width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width),
                                      nn.ReLU(inplace=True))
        self.up_conv3 = nn.Sequential(nn.Conv1d(2 * width, width // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width // 2),
                                      nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv1d(width // 2, width // 2, kernel_size=1, bias=False),
                                  nn.BatchNorm1d(width // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(width // 2, num_classes, kernel_size=1, bias=True))

    def forward(self, end_points):

        """
        upsampling, concat and 1x1conv (i.e. more points but less channels), contrary to encoding strategy, npoints=[4096,1152,304,88]
        these four stages are similar; up0 is slightly diff from up_{1,2,3} in channel number pattern while up_{1,2,3} follow same pattern
        - up0;res5 88x{3,1,2304=16C} res4 304_{3,1,1152=8C}; up(res5, res4)->304x2304 then concat res4 features-->304x3456->304x576=4C w. 1x1conv
        - up1;res4 304{3,1,1152=8C} res3 1152_{3,1,576=4C}; up(res4, res4)->1152x576 then concat res3 features-->1152x1152=8C->1152x288=2C w. 1x1conv
        - up2;...4096x144
        - up3;...15000x72
        - head;15000x72 -->1x1-->x1x-->15000xm where m is #classes.
        """
        
        # for clarity, assume npoints=[N, N/4, N/16, N/64, N/256] , C=144. of course, npoints can be like[4096,1152,304,88]
        # upsampling with masking, (B,16C,N/256) --> (B,16C,N/64)
        features = self.up0(end_points['res4_xyz'], end_points['res5_xyz'],
                            end_points['res4_mask'], end_points['res5_mask'], end_points['res5_features'])
        # shortcut connection, (B,16C,N/64)+(B,8C,N/64)=(B,24C,N/64)
        features = torch.cat([features, end_points['res4_features']], 1) # short-cut connection (simlar to unet's copy and paste, channel concat)
        # (B, 24C, N/64)-> (B, 4C, N/64)
        features = self.up_conv0(features)

        # KEY: the shape of output features of this upsample op is determined by feature channels(here different from the above channels)   
        # upsampling with masking, (B,4C,N/64) --> (B,4C,N/16), e.g., (2, 576,1152)
        features = self.up1(end_points['res3_xyz'], end_points['res4_xyz'],
                            end_points['res3_mask'], end_points['res4_mask'], features)
        # shortcut connection, (B,4C,N/16)+(B,4C,N/16) = (B,8C,N/16)
        features = torch.cat([features, end_points['res3_features']], 1)
        # (B, 8C, N/16)-> (B, 2C, N/16)
        features = self.up_conv1(features)

        # upsampling with masking, (B,2C,N/16) --> (B,2C,N/4)
        features = self.up2(end_points['res2_xyz'], end_points['res3_xyz'],
                            end_points['res2_mask'], end_points['res3_mask'], features)
        # shortcut connection, (B,2C,N/4)+(B,2C,N/4) = (B,4C,N/4)
        features = torch.cat([features, end_points['res2_features']], 1)
        # (B, 4C, N/4)-> (B, C, N/4)
        features = self.up_conv2(features)

        # upsampling with masking, (B,C,N/4) --> (B,C,N)
        features = self.up3(end_points['res1_xyz'], end_points['res2_xyz'],
                            end_points['res1_mask'], end_points['res2_mask'], features)
        # shortcut connection, (B,C,N)+(B,C,N) = (B,2C,N)
        features = torch.cat([features, end_points['res1_features']], 1)
        # (B, 2C, N)-> (B, C/2, N)
        features = self.up_conv3(features)

        # (B, C/2, npoints[i+3])-> (B, C/2, npoints[i+3])->(B,num_classes,npoints[i+3])
        logits = self.head(features)

        return logits
