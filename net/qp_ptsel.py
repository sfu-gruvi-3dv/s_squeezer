from core_io.serialize import dump_pickle
from typing import List, Tuple
from numpy.lib.arraysetops import isin

import torch
from torch import tensor
from core_io.meta_io import *
from torch_scatter import scatter
from einops import rearrange, repeat, asnumpy

from exp.scene_sq_utils import move_to_origin, normalize_3dpts
from net.pt_transformer import *
from net.qp_layer_cholesky import get_qp_layer
from core_math.cvxpy_qp_solver import CVXPY_QP, solve_qp_np

import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter

from matcher.superglue_matcher import BaseMatcher, SuperGlueMatcher
from dataset.common.hloc_db import Pt2dObs, Pt3dObs
from SuperGluePretrainedNetwork.models.superglue import normalize_keypoints

import torch.nn.functional as F
import time

import core_3dv.camera_operator_gpu as cam_opt_gpu
from core_dl.torch_ext import batch_sel_3d
from dataset.common.base_data_source import ClipMeta, Pt2dObs, Pt3dObs
from core_math.matrix_sqrt import sqrt_newton_schulz_autograd
from core_math.cvxpy_qp_solver import CVXPY_QP, solve_qp_np

def compute_rbf_kernel(in_feats, rbf_sigma=1.0):
    in_feats = in_feats.contiguous()
    pairwise_dist = torch.cdist(
        in_feats.unsqueeze(0), in_feats.unsqueeze(0), compute_mode='donot_use_mm_for_euclid_dist'
    ).squeeze(0)
    return torch.exp(-pairwise_dist / 2 * (rbf_sigma ** 2))

class PointDistanceKernel(nn.Module):
    def __init__(self, args: dict, in_ch=256):
        super(PointDistanceKernel, self).__init__()
        self.args = args
        self.pt_transformer = BasePointTransformer(dim=[in_ch, 256, 256, 128, 128], output_dim=64)

        # parameters
        self.move_to_origin = from_meta(self.args, 'move_to_origin', True)

    def forward(self, in_ref_xyz: torch.Tensor, in_ref_feats: torch.Tensor):
        """

        Args:
            in_ref_xyz: dim: (M, 3)
            in_ref_feats: dim: (M, C)

        Returns:

        """
        cur_dev = torch.cuda.current_device()

        r_xyz = in_ref_xyz.to(cur_dev)
        if self.move_to_origin:
            r_xyz = move_to_origin(r_xyz)
            r_xyz = normalize_3dpts(r_xyz)

        out_feat = self.pt_transformer(
            (r_xyz.unsqueeze(0), torch.cat([in_ref_feats, r_xyz.unsqueeze(0)], dim=-1).to(cur_dev))
        )
        out_feat = rearrange(out_feat, '() c n -> n c')
        out_feat = F.normalize(out_feat, dim=-1)

        # the out-product
        # TODO: this does redundant calculations for upper and lower triangles
        # find if this can be made more efficient
        outer_product = compute_rbf_kernel(out_feat, rbf_sigma=1.0)

        return outer_product, out_feat


class PointSelection(nn.Module):
    def __init__(self, args: dict, debug=False, solver_type='cvxpylayer'):
        super(PointSelection, self).__init__()
        self.args = args

        # rbf kernel
        self.dist_kernel_rbf_sigma = from_meta(args, 'dist_kernel_rbf_sigma', default=2.0)

        # qp layer
        self.qp_num_pts = from_meta(args, 'qp_num_pts', default=500)
        self.qp_compression_ratio = from_meta(args, 'qp_compression_ratio', default=0.5)
        self.qp_sqrtm_iters = from_meta(args, 'qp_sqrtm_iters', default=15)
        self.qp_solver_max_iters = from_meta(args, 'qp_solver_max_iters', default=3000)
        self.qp_solver_eps = from_meta(args, 'qp_solver_eps', default=1e-6)
        self.solver_type = solver_type
        if self.solver_type == 'cvxpylayer':
            self.qp_layer = get_qp_layer(self.qp_num_pts, self.qp_compression_ratio)
        elif self.solver_type == 'cvxpy':
            self.qp_layer = CVXPY_QP(self.qp_num_pts, self.qp_compression_ratio)

        # weight: lambda
        init_lambda_w = from_meta(args, 'qp_distinctiveness_weight', default=1.0)
        self.tao_o = Parameter(torch.ones(1) * init_lambda_w, requires_grad=True)
        self.register_parameter("tao_o", self.tao_o)
        print(self.tao_o)

    def get_pt_sel_thres(self):
        return 1 / (self.qp_num_pts * self.qp_compression_ratio + 1e-6)

    def get_distance_kernel(self, in_ref_xyz, normalize=True):

        if normalize:
            in_ref_xyz = move_to_origin(in_ref_xyz)
            in_ref_xyz = normalize_3dpts(in_ref_xyz)

        pairwise_dist = torch.cdist(
            in_ref_xyz.unsqueeze(0), in_ref_xyz.unsqueeze(0), compute_mode='donot_use_mm_for_euclid_dist'
        ).squeeze(0)
        return torch.exp(-pairwise_dist / 2 * (self.dist_kernel_rbf_sigma ** 2))

    def get_kernel(self, in_ref_xyz: torch.Tensor, in_kernel_feats: torch.Tensor, spatial_only=False):
        cur_dev = torch.cuda.current_device()
        assert in_ref_xyz.shape[0] == in_kernel_feats.shape[0]
        N, C = in_kernel_feats.shape

        fixed_kernel = self.get_distance_kernel(in_ref_xyz.to(cur_dev)).detach()
        # if spatial_only:
            # return fixed_kernel

        learnt_kernel = compute_rbf_kernel(in_kernel_feats.to(cur_dev).contiguous(), rbf_sigma=1.0)
        return 0.5 * (learnt_kernel + fixed_kernel)

    def sel_by_qp(self, kernel, dist_score, sel_idx=None):
        N = dist_score.view(-1).shape[0]
        kernel = kernel.view(N, N)
        if isinstance(sel_idx, np.ndarray):
            sel_idx = torch.from_numpy(sel_idx).to(dist_score.device)

        if sel_idx is not None:
            sel_dst_score = dist_score[sel_idx].clone()

            sel_kernel = kernel[sel_idx, :]
            sel_kernel = sel_kernel[:, sel_idx]
        else:
            sel_dst_score, sel_kernel = dist_score, kernel

        S = sel_kernel.shape[0]
        tao = torch.exp(self.tao_o)
        


        try:
            if self.solver_type == 'cvxpylayer':
                sel_kernel_sqrt, _ = sqrt_newton_schulz_autograd(sel_kernel.unsqueeze(0),
                                                                 numIters=self.qp_sqrtm_iters,
                                                                 dtype=sel_kernel.dtype)
                alpha_ = self.qp_layer(sel_kernel_sqrt.view(S, S), tao * sel_dst_score.view(S, 1),
                                       solver_args={"max_iters": self.qp_solver_max_iters,
                                                    "eps": self.qp_solver_eps})[0]
            else:
                alpha_ = solve_qp_np(sel_kernel.view(S, S), sel_dst_score.view(S, 1), compression_ratio=self.qp_compression_ratio, lambda_=tao.item())
                alpha_ = torch.from_numpy(alpha_).to(sel_dst_score.device).float()
                # alpha_ = self.qp_layer(sel_kernel.view(S, S), tao * sel_dst_score.view(S, 1))
        except Exception:
            return None

        # pad the discarded points' alphas with zeros
        if sel_idx is not None:
            qp_soln_padded = torch.zeros_like(dist_score)
            qp_soln_padded[sel_idx] = alpha_.view(-1)

        return qp_soln_padded

    def get_qp_energy(self, kernel, dist_score, alpha, sel_idx=None):

        if sel_idx is not None:
            sel_dist_score = dist_score.view(-1)[sel_idx]
            sel_kernel = kernel[sel_idx, :]
            sel_kernel = sel_kernel[:, sel_idx]
            sel_alpha = alpha.view(-1)[sel_idx]
        else:
            sel_dist_score, sel_kernel, sel_alpha = dist_score, kernel, alpha

        M = sel_kernel.shape[0]
        quad_form = sel_alpha.view(1, M) @ sel_kernel.view(M, M) @ sel_alpha.view(M, 1)
        linear_form = sel_alpha.view(1, M) @ sel_dist_score.view(M, 1)
        return quad_form - torch.exp(self.tao_o) * linear_form, quad_form, linear_form
