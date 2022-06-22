from pickle import TRUE
import numpy as np
import torch
import torch.nn as nn

import cv2
import random

from core_io.meta_io import from_meta
import core_3dv.camera_operator_gpu as cam_opt_gpu
from dataset.common.base_data_source import ClipMeta, Pt2dObs
from dataset.common.gt_corres_torch import corres_pos_from_pairs
from einops import asnumpy
from core_dl.expr_ctx import ExprCtx
from net.loss import DiffBinarizer
from core_3dv.weighted_dlt_pnp import DLT_P, DLT_RT


def compute_selected_inliers(rl2q_2d_err, r_sel_alpha, alpha_thres=0.5, reproj_inlier_thres=5):
    valid_refs = torch.where(r_sel_alpha > alpha_thres)[0]
    valid_rl2q_2d_err = rl2q_2d_err[valid_refs]
    if valid_rl2q_2d_err.shape[0] == 0:
        return 0, None
    return torch.sum(valid_rl2q_2d_err < reproj_inlier_thres), torch.sum(valid_rl2q_2d_err < reproj_inlier_thres) / valid_rl2q_2d_err.shape[0]


class DLTPnPLoss(nn.Module):

    """ randomly selects sets of 4-points to calculate PnP loss
    """
    def __init__(self, args: dict):
        super(DLTPnPLoss, self).__init__()
        self.args = args

        self.dlt_inlier_proj_thres = from_meta(args, 'dlt_inlier_proj_thres', default=14.0)
        self.full_repj_loss = from_meta(args, 'full_repj_loss', default=False)
        self.max_repj_err = from_meta(args, "max_rpj_err", default=40.0)


    def forward(self, r_xyz: torch.Tensor, r_alpha: torch.Tensor,
                q_pos_2d: torch.Tensor, q_K: torch.Tensor, q_gt_Tcw: torch.Tensor, q_dim_hw: tuple,
                r2q_matches
                ) -> dict:

        cur_dev = torch.cuda.current_device()
        r_alpha_b = r_alpha.to(cur_dev)

        # re-project the references points to query frame given gt query pose
        rl_gt_pos2d, rl_gt_depth = cam_opt_gpu.reproject(q_gt_Tcw, q_K, r_xyz)           # (M, 2), (M, )
        rl_valid = cam_opt_gpu.is_in_t(rl_gt_pos2d, rl_gt_depth, q_dim_hw)               # boolean: (M, )

        # filtering r2q_matches
        r2q_valid_flags = rl_valid[r2q_matches[:, 0]]
        r2q_valid_idx = torch.where(r2q_valid_flags == True)[0]
        r2q_matches = r2q_matches[r2q_valid_idx]

        # compute the err of r2q with gt pose
        rl_sel_pos2d, q_sel_pos2d = corres_pos_from_pairs(rl_gt_pos2d, q_pos_2d, r2q_matches)
        rl2q_2d_err = torch.norm(rl_sel_pos2d - q_sel_pos2d, dim=1)

        # outlier loss
        outlier_loss = rl2q_2d_err * r_alpha_b[r2q_matches[:, 0]]
        
        # extract 3D-2D correspondences
        r_sel_pos3d, q_sel_pos2d = corres_pos_from_pairs(r_xyz, q_pos_2d, r2q_matches)
        r_sel_alpha = r_alpha_b[r2q_matches[:, 0]]
        num_inliers, sel_inlier_ratios = compute_selected_inliers(rl2q_2d_err, r_sel_alpha, 
                                                                  reproj_inlier_thres=self.dlt_inlier_proj_thres)

        # condition check ----------------------------------------------------------------------------------------------
        if sel_inlier_ratios is None or num_inliers < 10:
            return None, None, None, None        

        # select the inliers for computing poses
        rl2q_2d_err_mask = (rl2q_2d_err < self.dlt_inlier_proj_thres)
        
        # compute inliers
        sel_pts = torch.where(r_sel_alpha > 0.5)[0]
        inliers = (rl2q_2d_err[sel_pts] < self.dlt_inlier_proj_thres)

        if rl2q_2d_err_mask.sum() > 6 and inliers.shape[0] > 16:
            
            if not self.full_repj_loss:
                est_P = DLT_P(q_sel_pos2d[rl2q_2d_err_mask], r_sel_pos3d[rl2q_2d_err_mask], r_sel_alpha[rl2q_2d_err_mask])
                r_prj_pos2d, _ = cam_opt_gpu.reproject(est_P, torch.eye(3).to(cur_dev), r_sel_pos3d[rl2q_2d_err_mask])
                r2q_repj_err = torch.norm(r_prj_pos2d - q_sel_pos2d[rl2q_2d_err_mask], dim=1)
            else:
                est_P = DLT_P(q_sel_pos2d, r_sel_pos3d, r_sel_alpha)
                r_prj_pos2d, _ = cam_opt_gpu.reproject(est_P, torch.eye(3).to(cur_dev), r_sel_pos3d)
                r2q_repj_err = torch.norm(r_prj_pos2d - q_sel_pos2d, dim=1)
            
            r2q_repj_err[r2q_repj_err > self.max_repj_err] = self.max_repj_err        
        
            print('Inliers: %.2f, NumInliers: %d, ReprojErr: %.2f' %
                  (sel_inlier_ratios, inliers.shape[0], r2q_repj_err.mean().item()))

        else:
            inlier_ratios = 0
            r2q_repj_err = torch.zeros(1, dtype=outlier_loss.dtype, device=outlier_loss.device)

        return outlier_loss, r2q_repj_err, rl2q_2d_err, sel_inlier_ratios

