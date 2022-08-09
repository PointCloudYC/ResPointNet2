import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

try:
    import pt_custom_ops._ext as _ext
except ImportError:
    raise ImportError(
        "Could not import _ext module.\n"
        "Please see the setup instructions in the README: "
        "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
    )

"""
CUDA ops linked the CUDA code
- GroupingOperation
- MaskedOrderedBallQuery
- MaskedNearestQuery
- MaskedGridSubsampling
- ThreeNN
- ThreeInterpolate
"""
class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class MaskedOrderedBallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, query_xyz, support_xyz, query_mask, support_mask):
        inds, inds_mask = _ext.masked_ordered_ball_query(query_xyz, support_xyz, query_mask,
                                                         support_mask, radius, nsample)
        ctx.mark_non_differentiable(inds, inds_mask)
        return inds, inds_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None


masked_ordered_ball_query = MaskedOrderedBallQuery.apply


class MaskedNearestQuery(Function):
    @staticmethod
    def forward(ctx, query_xyz, support_xyz, query_mask, support_mask):
        inds, inds_mask = _ext.masked_nearest_query(query_xyz, support_xyz, query_mask, support_mask)
        ctx.mark_non_differentiable(inds, inds_mask)
        return inds, inds_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


masked_nearest_query = MaskedNearestQuery.apply


class MaskedGridSubsampling(Function):
    @staticmethod
    def forward(ctx, xyz, mask, npoint, sampleDl):
        sub_xyz, sub_mask = _ext.masked_grid_subsampling(xyz, mask, npoint, sampleDl)  # B N 3

        ctx.mark_non_differentiable(sub_xyz, sub_mask)
        return sub_xyz, sub_mask

    @staticmethod
    def backward(xyz, a=None):
        return None, None, None, None


masked_grid_subsampling = MaskedGridSubsampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown(e.g., a query point) in known(a set of points w. known features). Note: suitable for SQN query network
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()

three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, C, m) point features to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)

three_interpolate = ThreeInterpolate.apply



"""
torch modules based on the above cuda ops
- MaskedQueryAndGroup, query npoint points w.r.t support_xyz with ball query strategy
- MaskedNearestQueryAndGroup, query npoint points w.r.t support_xyz with KNN strategy
- MaskedMaxPool, downsamping points w. max pool
- MaskedUpsample, upsamping points w. nearest or max strategy
- TrilinearInterpolate, trilinear interpolate features
"""
class MaskedQueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False):
        """query points(query_xyz) w.r.t. support points(support_xyz) within the radius with nsample points; Note: rely on masked_ordered_ball_query and group_operation CUDA ops.

        Args:
            radius ([type]): the search radius for ball query
            nsample ([type]): the number of nb points for a local region
            use_xyz (bool, optional): use xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): retain grouped xyz. Defaults to False.
            normalize_xyz (bool, optional): normalize xyz coordinates. Defaults to False.
        """
        super(MaskedQueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, features=None):
        """

        Args:
            query_xyz (tensor): queried points, (B,3,npoint)
            support_xyz (tensor): support points, (B,3,N)
            query_mask (tensor): mask
            support_mask (tensor): mask
            features (tensor): support features, (B,C,N). Defaults to None.

        Returns:
            new_features(tensor): grouped point features for each centroid points, (B, 3+C, npoint, npsample)
            idx_mask(tensor): mask, (B,npoint,nsample)
        """

        # (B,npoint,nsample), (B,npoint,nsample)
        idx, idx_mask = masked_ordered_ball_query(self.radius, self.nsample, query_xyz, support_xyz,
                                                  query_mask, support_mask)

        xyz_transpose = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_transpose, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1) # (B, 3, npoint, nsample), relative coordinates

        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None: # (B, C_feat, npoint)
            grouped_features = grouping_operation(features, idx)# (B, C, npoint, nsample)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz, idx_mask # (B, C + 3 or 3, npoint, nsample), (B,3,npoint,nsample) (B,npoint,nsample)
        else:
            return new_features, idx_mask # (B, C + 3 or 3, npoint, nsample), (B,npoint,nsample)


class MaskedNearestQueryAndGroup(nn.Module):
    def __init__(self, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False):
        """query points(query_xyz) w.r.t. support points(support_xyz) nearest neighboring points(e.g., K=3); base on masked_nearest_query and group_opeartion CUDA ops. Note: the logic is similar to MaskedQueryAndGroup class.

        Args:
            use_xyz (bool, optional): use xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): retain grouped xyz. Defaults to False.
            normalize_xyz (bool, optional): normalize xyz coordinates. Defaults to False.
        """
        super(MaskedNearestQueryAndGroup, self).__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, query_xyz, support_xyz, query_mask, support_mask, features=None):
        """the logic is similar to MaskedQueryAndGroup class, but differentiate on the query strategy (KNN)

        Args:
            query_xyz (tensor): queried points, (B,3,npoint)
            support_xyz (tensor): support points, (B,3,N)
            query_mask (tensor): mask
            support_mask (tensor): mask
            features (tensor): support features, (B,C,N). Defaults to None.

        Returns:
            new_features(tensor): grouped point features for each centroid points, (B, 3+C, npoint, npsample), nsample=1
            idx_mask(tensor): mask, (B,npoint,nsample)
        """

        # (B,npoint,nsample), (B,npoint,nsample)
        idx, idx_mask = masked_nearest_query(query_xyz, support_xyz, query_mask, support_mask)

        xyz_transpose = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_transpose, idx)  # (B, 3, npoint, 1)
        grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz, idx_mask
        else:
            return new_features, idx_mask # (B, C + 3 or 3, npoint, nsample=1), (B,npoint,nsample=1)


class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, nsample, sampleDl):
        """apply max pooling with masking over local region(determined by npoint, radius, nsample and sampledl), ending up with reduced points but with increased channels extract from local neighborhood

        Args:
            npoint ([type]): the number of points
            radius ([type]): the search radius for ball query
            nsample ([type]): the number of nb points for a local region
            sampleDl ([type]): voxel grid size(i.e. grid sub-sampling minimal sampling distance)
        """
        super(MaskedMaxPool, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.sampleDl = sampleDl
        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True)

    def forward(self, xyz, mask, features):

        # sub-sample, (B,3,npoint), (B,npoint)
        sub_xyz, sub_mask = masked_grid_subsampling(xyz, mask, self.npoint, self.sampleDl)
        sub_xyz = sub_xyz.contiguous()
        sub_mask = sub_mask.contiguous()

        # masked ordered ball query, (B, C, npoint, nsample), _, _
        neighborhood_features, grouped_xyz, idx_mask = self.grouper(sub_xyz, xyz, sub_mask, mask,
                                                                    features)  # (B, C, npoint, nsample)

        # apply max pooling to reduce/select features in the local region
        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )  # (B, C, npoint, 1)

        sub_features = torch.squeeze(sub_features, -1)  # (B, C, npoint)

        return sub_xyz, sub_mask, sub_features


class MaskedUpsample(nn.Module):
    def __init__(self, radius, nsample, mode='nearest'):
        """[summary]

        Args:
            radius ([type]): the search radius for ball query
            nsample ([type]): the number of nb points for a local region
            sampleDl ([type]): voxel grid size(i.e. grid sub-sampling minimal sampling distance)
            mode (str, optional): nearest or max. Defaults to 'nearest'.
        """
        super(MaskedUpsample, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mode = mode
        if mode == 'nearest':
            self.grouper = MaskedNearestQueryAndGroup(use_xyz=False, ret_grouped_xyz=True)
        else:
            self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True)

    def forward(self, up_xyz, xyz, up_mask, mask, features):
        """

        Args:
            up_xyz ([type]): up-sampled xyz, (B,3,N1), note: N1>=N2
            xyz ([type]): query xyz, (B,3,N2)
            up_mask ([type]): mask
            mask ([type]): mask
            features ([type]): query features, (B,C,N2)

        Raises:
            NotImplementedError: [description]

        Returns:
            up_feature(tensor): up-sampled features, (B,C,N1)
        """

        # nb features, (B, C, N1, nsample)
        neighborhood_features, grouped_xyz, idx_mask = self.grouper(up_xyz, xyz, up_mask, mask, features)  

        if self.mode == 'nearest':
            up_feature = neighborhood_features[..., 0].contiguous() # (B,C,N1)
        elif self.mode == 'max':
            up_feature = F.max_pool2d(neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]])
            up_feature = torch.squeeze(up_feature, -1) # (B,C,N1)
        else:
            raise NotImplementedError(f"mode:{self.mode} not supported in MaskedUpsample")
        return up_feature # (B,C,N1)


class TrilinearInterpolate(nn.Module):
    """Propagates the features of one set to another using trilinear interpolation

    Parameters
    ----------
    num_neighbors, the number of neighbors
    """

    def __init__(self, num_neighbors=3, ret_unknown_features=False):
        super(TrilinearInterpolate, self).__init__()
        self.num_neighbors=num_neighbors
        self.ret_unknown_features=ret_unknown_features
        # self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats=None, known_feats=None):
        """
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propagated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propagated

        Returns
        -------
        new_features : torch.Tensor
            (B, C2, n) tensor of the features of the unknown features
        """

        if known is not None:
            # (B,n,3), (B,n,3)
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8) # (B,n,3)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(
                known_feats, idx, weight
            )
        else:
            raise ValueError('make sure the known parameters are valid')

        if self.ret_unknown_features:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats # (B,C2,n)

        return new_features

class TrilinearInterpolateSQN(nn.Module):
    """Propagates the features of one set to another using trilinear interpolation for SQN

    Parameters
    ----------
    num_neighbors, the number of neighbors
    """

    def __init__(self, num_neighbors=3, ret_unknown_features=False):
        super(TrilinearInterpolateSQN, self).__init__()
        self.num_neighbors=num_neighbors
        self.ret_unknown_features=ret_unknown_features
        # self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, batch_inds, unknow_feats=None, known_feats=None):
        """
        Parameters
        ----------
        unknown : torch.Tensor
            (n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        batch_inds: torch.Tensor
            (B,) tensor of the batch indices to denote which batch for the unknown, values are 0 to B-1
        unknow_feats : torch.Tensor
            (C1, n) tensor of the features to be propagated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propagated

        Returns
        -------
        new_features : torch.Tensor
            (n, C2, 1) tensor of the features of the unknown features
        """

        # 转化思想，（n,3) 看成是n个batch的set，每个set只有1个点
        unknown = torch.reshape(unknown,(unknown.shape[0],1,-1)) # (n,1,3)
        # known should be (n, 15000, 3)
        known = known[batch_inds,...] # (n, 15000, 3)
        # known_feats should be (n, C2, 15000)
        known_feats = known_feats[batch_inds,...] # (n, C2, 15000)

        if known is not None:
            # (B,n,3), (B,n,3)
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8) # (B,n,3)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = three_interpolate(
                known_feats, idx, weight
            )
        else:
            raise ValueError('make sure the known parameters are valid')

        if self.ret_unknown_features:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats # (n,C2,1)

        return new_features