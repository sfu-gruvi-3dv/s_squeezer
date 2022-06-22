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
from evaluator.trajectory_eval import rel_distance, rel_R_deg


def compute_selected_inliers(rl2q_2d_err, r_sel_alpha, alpha_thres=0.5, reproj_inlier_thres=5):
    valid_refs = torch.where(r_sel_alpha > alpha_thres)[0]
    valid_rl2q_2d_err = rl2q_2d_err[valid_refs]
    if valid_rl2q_2d_err.shape[0] == 0:
        return 0, None
    return torch.sum(valid_rl2q_2d_err < reproj_inlier_thres), torch.sum(valid_rl2q_2d_err < reproj_inlier_thres) / valid_rl2q_2d_err.shape[0]




class FastPnPLoss(nn.Module):
    """ randomly selects sets of 4-points to calculate PnP loss
    """
    def __init__(self, args: dict, pt_sel_thres: float):
        super(FastPnPLoss, self).__init__()
        self.args = args
        self.pt_sel_thres = pt_sel_thres
        self.num_samples = from_meta(args, 'pnp_loss_samples', default=256)
        self.num_samples_qp_match = from_meta(args, 'pnp_samples_qp_match', default=2)
        self.sample_selected_pts = from_meta(args, 'sample_selected_pts', default=False)
        self.repj_to_query = from_meta(args, 'pnp_repj_to_query', default=True)
        self.inlier_thres = from_meta(args, 'pnp_inlier_thres', default=12)

        # make it smoother for the QP solution for better computing gradient
        # self.binarizer = DiffBinarizer(init_threshold=0.5, k=20.0)

    @staticmethod
    def sampling(alpha, num_samples=256, num_full_alpha_ratio=0.6, min_samples_qp_match=1):
        S = alpha.shape[0]
        r_sel_alpha_idx = asnumpy(torch.where(alpha > 0.5)[0])
        
        hyp_selpt_ids = []
        for x in range(num_samples):

            if x < num_samples * num_full_alpha_ratio:
                sel_pt_ids = np.random.choice(r_sel_alpha_idx, 4)
            else:
                while True:
                    sel_pt_ids = random.sample(range(S), 4)
                    # at least self.num_samples_qp_match point in triplet should be selected by QP
                    if (alpha[sel_pt_ids] > 0.3).sum().item() >= min_samples_qp_match:
                        break
            hyp_selpt_ids.append(sel_pt_ids)
            
        hyp_selpt_ids = torch.from_numpy(np.asarray(hyp_selpt_ids))
        return hyp_selpt_ids

    @staticmethod
    def pnp(pos2d, pos3d, K):
        H = pos2d.shape[0]
        if isinstance(K, torch.Tensor):
            K = asnumpy(K)
            
        # pnp
        h_Rts = []
        for h in range(H):
            hp_pos_2d = pos2d[h]
            hr_pos_3d = pos3d[h]
            
            p_success, rq, t = cv2.solvePnP(asnumpy(hr_pos_3d), asnumpy(hp_pos_2d), K.reshape(3, 3), None,
                                            flags=cv2.SOLVEPNP_P3P)
            if not p_success:
                h_Rts.append(np.eye(4)[:3, :].reshape(1, 3, 4))
                continue

            R, _ = cv2.Rodrigues(rq)
            Rt = np.eye(4).astype(np.float32)
            Rt[:3, :3], Rt[:3, 3] = R, t.ravel()
            h_Rts.append(Rt[:3, :].reshape(1, 3, 4))

        h_Rts = torch.from_numpy(np.vstack(h_Rts)).float()        
        return h_Rts.to(pos2d.device)
    
    @staticmethod
    def hypo_pose_err(hypo_pose: torch.Tensor, gt_pose: torch.Tensor):
        H = hypo_pose.shape[0]
        
        q_gt_Tcw_ = asnumpy(gt_pose)
        rot_deg_errs, trans_errs = [], []
        for h in range(H):
            h_Rt = asnumpy(hypo_pose[h])

            t_err, r_err = rel_distance(h_Rt, q_gt_Tcw_), rel_R_deg(h_Rt, q_gt_Tcw_)
            rot_deg_errs.append(r_err)
            trans_errs.append(t_err)
        rot_deg_errs = np.asarray(rot_deg_errs)
        trans_errs = np.asarray(trans_errs)        
        return rot_deg_errs, trans_errs
    
    @staticmethod
    def hypo_repj_errs(hypo_pose, local_uv, world_xyz, K):
        assert local_uv.shape[0] == world_xyz.shape[0]
        H = hypo_pose.shape[0]
        
        h_r_proj_pt2d, _ = cam_opt_gpu.reproject(hypo_pose,
                                                 K.view(1, 3, 3).expand(H, -1, -1).to(hypo_pose.device),
                                                 world_xyz.expand(H, -1, -1).to(hypo_pose.device))
        h_r2q_err = h_r_proj_pt2d - local_uv.expand(H, -1, -1).to(h_r_proj_pt2d.device)
        h_r2q_err = torch.norm(h_r2q_err, dim=2)
        return h_r2q_err

    def forward(self, r_xyz: torch.Tensor, r_alpha: torch.Tensor,
                q_pos_2d: torch.Tensor, q_K: torch.Tensor, q_gt_Tcw: torch.Tensor, q_dim_hw: tuple,
                r2q_matches
                ) -> dict:
        """
        @param xyz: (N, 3) Point coords
        @param r_alpha: (N,) Points distribution (alpha)
        """
        cur_dev = torch.cuda.current_device()
        r_alpha_b = r_alpha.to(cur_dev)

        # re-project the references points to query frame given gt query pose
        rl_gt_pos2d, rl_gt_depth = cam_opt_gpu.reproject(q_gt_Tcw, q_K, r_xyz)            # (M, 2), (M, )
        rl_valid = cam_opt_gpu.is_in_t(rl_gt_pos2d, rl_gt_depth, q_dim_hw)               # boolean: (M, )

        # filtering r2q_matches
        r2q_valid_flags = rl_valid[r2q_matches[:, 0]]
        r2q_valid_idx = torch.where(r2q_valid_flags == True)[0]
        r2q_matches = r2q_matches[r2q_valid_idx]

        # re-projection err between projected ref. (using GT query pose) to the matched query keypoint position
        rl_sel_pos2d, q_sel_pos2d = corres_pos_from_pairs(rl_gt_pos2d, q_pos_2d, r2q_matches)
        rl_sel_alpha = r_alpha[r2q_matches[:, 0]]
        rl_sel_pos3d = r_xyz[r2q_matches[:, 0]]
        rl2q_2d_err = torch.norm(rl_sel_pos2d - q_sel_pos2d, dim=1)
        rl2q_2d_err[rl2q_2d_err > 50.0] = 50.0

        outlier_loss = rl2q_2d_err * rl_sel_alpha
        
        # condition check ----------------------------------------------------------------------------------------------
        num_inliers, inlier_ratio = compute_selected_inliers(rl2q_2d_err, rl_sel_alpha, alpha_thres=0.3, reproj_inlier_thres=12)
        if num_inliers < 10 or inlier_ratio < 0.2:
            return outlier_loss, None, None, None

        # sampling -----------------------------------------------------------------------------------------------------        
        hyp_selpt_ids = self.sampling(rl_sel_alpha, num_samples=self.num_samples, min_samples_qp_match=2)

        hypo_alphas = rl_sel_alpha[hyp_selpt_ids.view(-1)].view(-1, 4)
        hypo_q_pos2d = q_sel_pos2d[hyp_selpt_ids.view(-1)].view(-1, 4, 2)
        hypo_r_pos3d = rl_sel_pos3d[hyp_selpt_ids.view(-1)].view(-1, 4, 3)

        # compute pose of each hypothesis
        h_Rts = self.pnp(hypo_q_pos2d, hypo_r_pos3d, q_K)
        
        # compute repj_
        if self.repj_to_query:
            h_repj_err = self.hypo_repj_errs(h_Rts, q_sel_pos2d, rl_sel_pos3d, q_K)
        else:
            # re-projection error using references points with ground-truth query pose
            r_valid_idx = torch.where(rl_valid == True)[0]
            r_valid_xyz = r_xyz[r_valid_idx, :].view(1, r_valid_idx.shape[0], 3)
            rl_valid_gt_pos2d = rl_gt_pos2d[r_valid_idx].view(1, -1, 2)
            
            h_repj_err = self.hypo_repj_errs(h_Rts, rl_valid_gt_pos2d, r_valid_xyz, q_K)
            
        h_repj_err[h_repj_err > 50] = 50.0
        
        # compute inliers
        h_r2q_inlier = h_repj_err < self.inlier_thres
        h_r2q_inlier_ratio = h_r2q_inlier.sum(dim=1) / h_r2q_inlier.shape[1]        

        # gather alphas from hypothesis
        h_m_alpha = torch.prod(hypo_alphas, dim=1)
        h_pnp_err = h_m_alpha * (1 - h_r2q_inlier_ratio.detach())

        return outlier_loss, h_pnp_err, rl2q_2d_err, inlier_ratio


    def evaluate(self, r_xyz: torch.Tensor, r_alpha: torch.Tensor, 
                q_pos_2d: torch.Tensor, q_K: torch.Tensor, q_gt_Tcw: torch.Tensor, q_dim_hw: tuple, r2q_matches):
        cur_dev = torch.cuda.current_device()
        r_alpha_b = r_alpha.to(cur_dev)

        # re-project the references points to query frame given gt query pose
        rl_gt_pos2d, rl_gt_depth = cam_opt_gpu.reproject(q_gt_Tcw, q_K, r_xyz)            # (M, 2), (M, )
        rl_valid = cam_opt_gpu.is_in_t(rl_gt_pos2d, rl_gt_depth, q_dim_hw)               # boolean: (M, )

        # filtering r2q_matches
        r2q_valid_flags = rl_valid[r2q_matches[:, 0]]
        r2q_valid_idx = torch.where(r2q_valid_flags == True)[0]
        r2q_matches = r2q_matches[r2q_valid_idx]

        # re-projection err between projected ref. (using GT query pose) to the matched query keypoint position
        rl_sel_pos2d, q_sel_pos2d = corres_pos_from_pairs(rl_gt_pos2d, q_pos_2d, r2q_matches)
        rl_sel_alpha = r_alpha[r2q_matches[:, 0]]
        rl_sel_pos3d = r_xyz[r2q_matches[:, 0]]
        rl2q_2d_err = torch.norm(rl_sel_pos2d - q_sel_pos2d, dim=1)
        rl2q_2d_err[rl2q_2d_err > 50.0] = 50.0

        outlier_loss = rl2q_2d_err * rl_sel_alpha
        
        # condition check ----------------------------------------------------------------------------------------------
        num_inliers, inlier_ratio = compute_selected_inliers(rl2q_2d_err, rl_sel_alpha)
        if num_inliers < 10 or inlier_ratio < 0.2:
            return outlier_loss, None, None, None

        # sampling -----------------------------------------------------------------------------------------------------        
        hyp_selpt_ids = self.sampling(rl_sel_alpha, num_samples=self.num_samples, min_samples_qp_match=2)

        hypo_alphas = rl_sel_alpha[hyp_selpt_ids.view(-1)].view(-1, 4)
        hypo_q_pos2d = q_sel_pos2d[hyp_selpt_ids.view(-1)].view(-1, 4, 2)
        hypo_r_pos3d = rl_sel_pos3d[hyp_selpt_ids.view(-1)].view(-1, 4, 3)

        # compute pose of each hypothesis
        h_Rts = self.pnp(hypo_q_pos2d, hypo_r_pos3d, q_K)
        
        # compute repj_
        if self.repj_to_query:
            h_repj_err = self.hypo_repj_errs(h_Rts, q_sel_pos2d, rl_sel_pos3d, q_K)
        else:
            # re-projection error using references points with ground-truth query pose
            r_valid_idx = torch.where(rl_valid == True)[0]
            r_valid_xyz = r_xyz[r_valid_idx, :].view(1, r_valid_idx.shape[0], 3)
            rl_valid_gt_pos2d = rl_gt_pos2d[r_valid_idx].view(1, -1, 2)
            
            h_repj_err = self.hypo_repj_errs(h_Rts, rl_valid_gt_pos2d, r_valid_xyz, q_K)
            
        h_repj_err[h_repj_err > 50] = 50.0
        
        # compute inliers
        h_r2q_inlier = h_repj_err < self.inlier_thres
        h_r2q_inlier_ratio = h_r2q_inlier.sum(dim=1) / h_r2q_inlier.shape[1]

        # compute camera pose err
        rot_err, trans_err = self.hypo_pose_err(h_Rts, q_gt_Tcw)

        return rot_err, trans_err, h_r2q_inlier_ratio
