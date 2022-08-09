import torch
import torch.nn as nn

try:
    from .backbones import ResNet
except:
    from backbones import ResNet
try:
    from .heads import ClassifierResNet, SceneSegHeadResNet, SqnHeadResNet
except:
    from heads import ClassifierResNet, SceneSegHeadResNet, SqnHeadResNet
try:
    from .losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy
except:
    from losses import LabelSmoothingCrossEntropyLoss, MultiShapeCrossEntropy, MaskedCrossEntropy


def build_scene_segmentation(config):
    """create a semantic segmentation model and the loss with mask

    Args:
        config (dict): the config object

    Returns:
        tuple: the model and loss
    """

    model = SceneSegmentationModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio)
    criterion = MaskedCrossEntropy()
    return model, criterion

def build_scene_segmentation_sqn(config,weak_supervision_ratio=0.1):
    """create a semantic segmentation model and the loss with mask

    Args:
        config (dict): the config object

    Returns:
        tuple: the model and loss
    """
    model = SqnModel(config, config.backbone, config.head, config.num_classes,
                                   config.input_features_dim,
                                   config.radius, config.sampleDl, config.nsamples, config.npoints,
                                   config.width, config.depth, config.bottleneck_ratio,weak_supervision_ratio=weak_supervision_ratio)
    # criterion = MaskedCrossEntropy()
    criterion = nn.CrossEntropyLoss()
    return model, criterion

class SceneSegmentationModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints,
                 width=144, depth=2, bottleneck_ratio=2):
        """Semantic segmentation model

        Args:
            config (dict): the config object
            backbone (str): resnet or ...
            head (str): resnet_scene_seg or ...
            num_classes (int): the number of classes
            input_features_dim (int): the dimension of input features
            radius (float): neighborhood search radius for local feature learning
            sampleDl (float): voxel grid size
            nsamples (list): the number of neighboring points for each residual stage, e.g., [26,31,...]
            npoints (list): the number of down-sampling points for each residual stage, e.g., [4096,1152,...]
            width (int, optional): the channels of the network. Defaults to 144.
            depth (int, optional): the block repeating factor. Defaults to 2.
            bottleneck_ratio (int, optional): the bottleneck division ratio. Defaults to 2.

        Raises:
            NotImplementedError: backbone not yet implemented
            NotImplementedError: head not yet implemented
        """
        super(SceneSegmentationModel, self).__init__()
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'resnet_scene_seg':
            self.segmentation_head = SceneSegHeadResNet(num_classes, width, radius, nsamples)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Multi-Part Segmentation Model")

    def forward(self, xyz, mask, features):

        # end_points is a dict containing {xyz,features,mask} for all stages(e.g., 5 stages)
        end_points = self.backbone(xyz, mask, features)
        return self.segmentation_head(end_points)

    def init_weights(self):

        # if conv2d or conv1d, then only init weights w. kaiming_normal_ and set biases = zero due to bach normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight) # in place method
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias) # in place method


class SqnModel(nn.Module):
    def __init__(self, config, backbone, head, num_classes, input_features_dim, radius, sampleDl, nsamples, npoints, width=144, depth=2, bottleneck_ratio=2, weak_supervision_ratio=0.1):
        """SQN Semantic segmentation model

        Args:
            config (dict): the config object
            backbone (str): resnet or ...
            head (str): resnet_scene_seg or ...
            num_classes (int): the number of classes
            input_features_dim (int): the dimension of input features
            radius (float): neighborhood search radius for local feature learning
            sampleDl (float): voxel grid size
            nsamples (list): the number of neighboring points for each residual stage, e.g., [26,31,...]
            npoints (list): the number of down-sampling points for each residual stage, e.g., [4096,1152,...]
            width (int, optional): the channels of the network. Defaults to 144.
            depth (int, optional): the block repeating factor. Defaults to 2.
            bottleneck_ratio (int, optional): the bottleneck division ratio. Defaults to 2.
            weak_supervision_ratio(float, optional): the weakly supervision ratio, e.g., 0.1, 0.01 and 0.01.

        Raises:
            NotImplementedError: backbone not yet implemented
            NotImplementedError: head not yet implemented
        """
        super(SqnModel, self).__init__()
        self.weak_supervision_ratio = weak_supervision_ratio
        if backbone == 'resnet':
            self.backbone = ResNet(config, input_features_dim, radius, sampleDl, nsamples, npoints,
                                   width=width, depth=depth, bottleneck_ratio=bottleneck_ratio)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented in Multi-Part Segmentation Model")

        if head == 'sqn_head':
            self.segmentation_head = SqnHeadResNet(num_classes,weak_supervision_ratio=weak_supervision_ratio)
        else:
            raise NotImplementedError(f"Head {backbone} not implemented in Semantic Segmentation Model")

    def forward(self, xyz, mask, features, weakly_xyz, batch_inds):

        # end_points is a dict containing {xyz,features,mask} for all stages(e.g., 5 stages)
        end_points = self.backbone(xyz, mask, features)

        # shuffle the points and retrieve weak_supervision_ratio*N
        # N=end_points['res1_xyz'].shape[1]
        # permutation=torch.randperm(N)
        # selected_idx = permutation[:int(N*self.weak_supervision_ratio)]
        # weakly_xyz = xyz[:,selected_idx,:] # (B,N*weakly_ratio,3)
        # weakly_mask = mask[:,selected]

        return self.segmentation_head(weakly_xyz, end_points, batch_inds)

    def init_weights(self):
        # if conv2d or conv1d, then only init weights w. kaiming_normal_ and set biases = zero due to bach normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight) # in place method
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias) # in place method


if __name__ == "__main__":

    # obtain config
    import argparse
    import os
    import sys
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    sys.path.append(BASE_DIR)
    sys.path.append(ROOT_DIR)
    from utils.config import config, update_config
    parser = argparse.ArgumentParser('PSNet5 semantic segmentation training')
    parser.add_argument('--cfg', type=str, default='cfgs/psnet5/respoinet2_dp_fi_df_fc1_max.yaml', help='config file')
    args, unparsed = parser.parse_known_args()
    # update config dict with the yaml file
    update_config(args.cfg)
    print(config)

    # create a model and loss
    model, criterion = build_scene_segmentation(config)
    print(model)
    # IMPORTANT: place model to GPU so that be able to test GPU CUDA ops
    if torch.cuda.is_available():
        model=model.cuda()

    # create a random input and then predict
    batch_size = 2 # config.batch_size  
    num_points = config.num_points
    input_features_dim = config.input_features_dim
    # IMPORTANT: place these tensors to GPU so that be able to test GPU CUDA ops
    if torch.cuda.is_available():
        xyz = torch.rand(batch_size,num_points,3).cuda()
        mask= torch.ones(batch_size,num_points).cuda()
        features = torch.rand(batch_size,input_features_dim,num_points).cuda()
        labels = torch.ones(batch_size,num_points,dtype=torch.long).cuda()
    else:
        xyz = torch.rand(batch_size,num_points,3)
        mask= torch.ones(batch_size,num_points)
        features = torch.rand(batch_size,input_features_dim,num_points)
        labels = torch.ones(batch_size,num_points,dtype=torch.long)

    # predict
    preds = model(xyz, mask, features)
    print(preds.shape, preds)

    # compute loss
    loss = criterion(preds,labels,mask)
    print(loss)
