from typing import List, Tuple
from numpy.lib.arraysetops import isin
import torch
from torch import tensor
from core_io.meta_io import *
from torch_scatter import scatter
from einops import rearrange, repeat, asnumpy
# from net.pt_transformer import BasePointTransformer
import torch.nn as nn
import numpy as np
from matcher.superglue_matcher import BaseMatcher
from SuperGluePretrainedNetwork.models.superglue import normalize_keypoints
import torch.nn.functional as F
from net.pt_transformer import BasePointTransformer


def aggr_pc_feats(pt2d_feats_t, pt2d_obs3d_t, aggr_method='mean'):
    """
        Aggregate the point cloud features

    Args:
        pt2d_feats_t (Tensor): point 2d feature, dim (N, C)
        pt2d_obs3d_t (Tensor): the corresponding observed 3D point id, dim (N)
        aggr_method (str): the aggregate method

    Returns:
        aggregated features

    """
    out = scatter(pt2d_feats_t, pt2d_obs3d_t, dim=0, reduce=aggr_method)
    return out

def normalize_kpts_pos(img_shapes, kpts_pos):
    """
        Normalize the keypoint 2d position from -1 to 1
    Args:
        img_shapes (tuple or list): image dimensions
        kpts_pos (Tensor): the keypoint 2d positions (u, v)

    Returns:
        normalized keypoint position.

    """    
    img_dims = [(1, 3, int(shape[0].item()), int(shape[1].item())) for shape in img_shapes]
    return [normalize_keypoints(kpts, img_dims[i]) for i, kpts in enumerate(kpts_pos)]


def normalize_2d_pos(dim_hw: Tuple or List, kpt_2d_pos: torch.Tensor):
    if kpt_2d_pos.ndim == 3:
        kpt_2d_pos = kpt_2d_pos.view(-1, 2)
    return normalize_keypoints(kpt_2d_pos, (1, 3, dim_hw[0], dim_hw[1])).squeeze(0)


def move_to_origin(pt3d_xyz):
    """
        Move the point cloud to origin (aka, normalization)

    Args:
        pt3d_xyz (Tensor): point cloud xyz

    Returns: point cloud

    """
    return pt3d_xyz - pt3d_xyz.mean(dim=1).unsqueeze(0)


def normalize_pts(pt3d_xyz):
    scale_x = pt3d_xyz[:, 0].max() - pt3d_xyz[:, 0].min()
    scale_y = pt3d_xyz[:, 1].max() - pt3d_xyz[:, 1].min()
    scale_z = pt3d_xyz[:, 2].max() - pt3d_xyz[:, 2].min()
    max_dim = max(max(scale_x, scale_y), scale_z)
    pt3d_xyz /= (max_dim + 1e-5)
    return pt3d_xyz * 2


def q2r_2d_regs(matches_2d3d: torch.Tensor, r_info_dict: dict):
    """
        Convert the 2D to 3D correspondences to 2D to 2D correspondences of reference frame

    Args:
        matches_2d3d (Tensor): 2D to 3D correspondences (index)
        r_info_dict (dict): the reference info in dict.

    Returns:
        list of pairs of correspondences (2D to 2D)

    """
    match_np = asnumpy(matches_2d3d)

    match_q2r = dict()
    for r in range(len(r_info_dict['pt2d_obs3d'])):
        pt3d_obs3d_np = asnumpy(r_info_dict['pt2d_obs3d'][r][0])

        # build the local keypoint-pairs
        pairs = []
        for i, ref_pt_idx in enumerate(match_np[:, 1]):
            x = np.where(pt3d_obs3d_np == ref_pt_idx)[0]
            if len(x) == 1:
                pairs.append((match_np[i, 0], x[0]))

        match_q2r[r] = np.asarray(pairs)

    return match_q2r

def corres_pos_from_pairs(a_pos, b_pos, match_a2b):
    """
        Gather correspondences position (u,v) given match indices.

    Args:
        a_pos (Tensor or Array): the position in frame a
        b_pos (Tensor or Array): the position in frame b
        match_a2b (Array): the correspondences indices between a and b

    Returns:
        correspondences in 2d position (u, v) between a and b

    """
    a_kpt_pos = a_pos[match_a2b[:, 0], :]
    b_kpt_pos = b_pos[match_a2b[:, 1], :]
    return (a_kpt_pos, b_kpt_pos)

""" Scene Squeezer
"""
class SceneSqueezer(nn.Module):

    def __init__(self, args:dict, kypt_matcher: BaseMatcher):
        super(SceneSqueezer, self).__init__()
        self.args = args
        self.matcher = kypt_matcher
        self.pt_transformer = BasePointTransformer(dim=[256, 256, 256, 256, 512])
        self.relu = nn.ReLU()

        # parameters
        self.move_to_origin = from_meta(self.args, 'move_to_origin', True)

    def preprocess(self, input_data):

        # normalize the keypoint 2D position
        input_data['pt2d_pos_n'] = normalize_kpts_pos(input_data['dims'], input_data['pt2d_pos'])

        return input_data

    def forward(self, input_data: dict):
        cur_dev = torch.cuda.current_device()
        
        input_data = self.preprocess(input_data)

        with torch.no_grad():

            # encode features
            pt2d_npos_t = torch.cat([r.squeeze(0) for r in input_data['pt2d_pos_n']])
            pt2d_feats_t = torch.cat([r.squeeze(0) for r in input_data['pt2d_feats']])
            pt2d_scores_t = torch.cat([r.squeeze(0) for r in input_data['pt2d_scores']])

            pt2d_npos_t = rearrange(pt2d_npos_t, 'n v -> () n v')
            pt2d_feats_t = rearrange(pt2d_feats_t, 'n c -> () c n')
            pt2d_scores_t = rearrange(pt2d_scores_t, 'n -> () n')
            pt2d_feats_t = self.matcher.encode_pt_feats([pt2d_feats_t, pt2d_npos_t, pt2d_scores_t])

            # aggregate 2D features for each 3D point.
            pt2d_obs3d_t = torch.cat([r.squeeze(0) for r in input_data['pt2d_obs3d']])
            aggr_2d_feats = aggr_pc_feats(pt2d_feats_t.squeeze(0).transpose(0, 1), pt2d_obs3d_t)

        # get distinctive scores by forwarding with point transformer
        N, C = aggr_2d_feats.shape

        pt3d_xyz = input_data['pt3d'].to(cur_dev)
        if self.move_to_origin:
            pt3d_xyz = move_to_origin(pt3d_xyz)
            pt3d_xyz = normalize_pts(pt3d_xyz)

        aggr_2d_feats = rearrange(aggr_2d_feats, 'n c -> () n c').to(cur_dev)
        log_var = self.pt_transformer.forward((pt3d_xyz, aggr_2d_feats)).view(1, N)
        # log_var = F.softplus(log_var)
        
        # dist_scores = self.relu(dist_scores)
        # dist_scores = torch.sigmoid(dist_scores).view(N)

        return log_var, (pt3d_xyz, rearrange(aggr_2d_feats, 'b n c -> b c n'))
