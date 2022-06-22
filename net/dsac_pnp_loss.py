# from pickle import TRUE
# import numpy as np
# import torch
# from torch._C import dtype
# import torch.nn as nn

# import cv2
# import random

# from core_io.meta_io import from_meta
# import core_3dv.camera_operator_gpu as cam_opt_gpu
# from dataset.common.base_data_source import ClipMeta, Pt2dObs
# from dataset.common.gt_corres_torch import corres_pos_from_pairs
# from einops import asnumpy
# from core_dl.expr_ctx import ExprCtx
# from net.loss import DiffBinarizer
# from core_3dv.weighted_dlt_pnp import DLT_P, DLT_RT
# from einops import asnumpy, rearrange
# import torch.nn.functional as F


# def compute_selected_inliers(rl2q_2d_err, r_sel_alpha, alpha_thres=0.5, reproj_inlier_thres=5):
#     valid_refs = torch.where(r_sel_alpha > alpha_thres)[0]
#     valid_rl2q_2d_err = rl2q_2d_err[valid_refs]
#     if valid_rl2q_2d_err.shape[0] == 0:
#         return 0, None
#     return torch.sum(valid_rl2q_2d_err < reproj_inlier_thres), torch.sum(valid_rl2q_2d_err < reproj_inlier_thres) / valid_rl2q_2d_err.shape[0]


# class DSACPnPLoss(nn.Module):
#     """ randomly selects sets of 4-points to calculate PnP loss
#     """
#     def __init__(self, args: dict, verbose=True):
#         super(DSACPnPLoss, self).__init__()
#         self.args = args

#         # number of hypotheses
#         self.hypotheses = from_meta(args, 'dsac_hypotheses', default=256)

#         # inlier threshold in pixels 
#         self.reprj_inlier_thres = from_meta(args, 'dsac_reprj_inlier_thres', default=12)

#         # alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer
#         self.reprj_inlier_alpha = from_meta(args, 'dsac_reprj_inlier_alpha', default=10.0)

#         # beta parameter of the soft inlier count; controls the softness of the sigmoid; lower means softer
#         self.reprj_inlier_beta = from_meta(args, 'dsac_reprj_inlier_beta', default=0.5)

#         # maximum reprojection error; reprojection error is clamped to this value for stability
#         self.reprj_maxerr_thres = from_meta(args, 'dsac_reprj_maxerr_thres', default=0.5)
        
#         #  how many loops when training
#         self.train_sample_loops = from_meta(args, 'dsac_train_sample_loops', default=1)
        
#         # weights for loss function
#         self.w_loss_rot = from_meta(args, 'dsac_w_loss_rot', default=1.0)
#         self.w_loss_trans = from_meta(args, 'dsac_w_loss_trans', default=1.0)
        
#         self.verbose = verbose
        
#         # make it smoother for the QP solution for better computing gradient
#         self.binarizer = DiffBinarizer(init_threshold=0.5, k=35.0)
       
#     def forward(
#         self, r_xyz: torch.Tensor, r_alpha: torch.Tensor, 
#         q_pos_2d: torch.Tensor, q_K: torch.Tensor, q_gt_Tcw: torch.Tensor, q_dim_hw: tuple, r2q_matches: torch.Tensor):

#         cur_dev = torch.cuda.current_device()
#         r_alpha_b = r_alpha.to(cur_dev)
        
#         # re-project the references points to query frame given gt query pose
#         rl_gt_pos2d, rl_gt_depth = cam_opt_gpu.reproject(q_gt_Tcw, q_K, r_xyz)           # (M, 2), (M, )
#         rl_valid = cam_opt_gpu.is_in_t(rl_gt_pos2d, rl_gt_depth, q_dim_hw)               # boolean: (M, )

#         # filtering r2q_matches
#         r2q_valid_flags = rl_valid[r2q_matches[:, 0]]
#         r2q_valid_idx = torch.where(r2q_valid_flags == True)[0]
#         r2q_matches = r2q_matches[r2q_valid_idx]

#         # compute the err of r2q with gt pose
#         rl_sel_pos2d, q_sel_pos2d = corres_pos_from_pairs(rl_gt_pos2d, q_pos_2d, r2q_matches)
#         rl2q_2d_err = torch.norm(rl_sel_pos2d - q_sel_pos2d, dim=1)
        
#         # extract 3D-2D correspondences
#         r_sel_pos3d, q_sel_pos2d = corres_pos_from_pairs(r_xyz, q_pos_2d, r2q_matches)
#         r_sel_alpha = r_alpha_b[r2q_matches[:, 0]]
#         num_inliers, sel_inlier_ratios = compute_selected_inliers(rl2q_2d_err, r_sel_alpha, reproj_inlier_thres=self.reprj_inlier_thres)
            
#         # outlier loss
#         outlier_loss = rl2q_2d_err * r_alpha_b[r2q_matches[:, 0]]
        
#         # condition check ----------------------------------------------------------------------------------------------
#         if sel_inlier_ratios is None or num_inliers < 10:
#             return None, None, None, None

#         # dsac loss ----------------------------------------------------------------------------------------------------
#         q_gt_Tcw_inv = torch.eye(4)
#         R_inv, t_inv = cam_opt_gpu.camera_pose_inv(q_gt_Tcw[:3, :3], q_gt_Tcw[:3, 3])
#         q_gt_Tcw_inv[:3, :3] = R_inv
#         q_gt_Tcw_inv[:3, 3] = t_inv.view(-1)
#         q_K_ = asnumpy(q_K)
#         fx, fy, cx, cy = q_K_[0, 0], q_K_[1, 1], q_K_[0, 2], q_K_[1, 2]
        
#         local_uv = rearrange(q_sel_pos2d, '(B N W) D -> B D N W', B=1, W=1)
#         world_xyz = rearrange(r_sel_pos3d, '(B N W) D -> B D N W', B=1, W=1)
#         r_sel_alpha_w = rearrange(r_sel_alpha, '(B N W) -> B () N W', B=1, W=1)
#         r_sel_alpha_w = F.softmax(r_sel_alpha_w, dim=2)

#         alpha_w_grad = torch.zeros(r_sel_alpha.size(), device='cpu', requires_grad=False)
#         alpha_gradient_samples = []
#         loss_samples = []                
#         for i_loop in range(self.train_sample_loops):
#             out_refined_pose = torch.zeros(4, 4)
#             out_hyp_probs = torch.zeros(self.hypotheses)
#             out_hyp_loss = torch.zeros(self.hypotheses)
            
#             world_xyz_grad_ = torch.zeros_like(world_xyz, device='cpu', requires_grad=False)
#             alpha_w_grad_ = torch.zeros_like(r_sel_alpha_w, device='cpu', requires_grad=False)
#             e_loss = ngdsac.backward(local_uv.cpu(),
#                                      world_xyz.cpu(), world_xyz_grad_, r_sel_alpha_w.cpu(), alpha_w_grad_, 
#                                      q_gt_Tcw_inv, out_refined_pose, out_hyp_probs, out_hyp_loss,
#                                      0, 0,
#                                      self.hypotheses,
#                                      self.reprj_inlier_thres, 
#                                      fx, cx, cy, self.w_loss_rot, self.w_loss_trans, self.reprj_inlier_alpha,
#                                      self.reprj_inlier_beta, self.reprj_maxerr_thres, 1, i_loop, self.verbose)
#             hyp_var = out_hyp_probs.var()
#             print(hyp_var)
#             loss_samples.append(e_loss)
#             alpha_gradient_samples.append(alpha_w_grad_.view(-1))

#         baseline = sum(loss_samples) / self.train_sample_loops
        
#         # substract baseline and calculte gradients
#         for i, l in enumerate(loss_samples):
#             alpha_w_grad += alpha_gradient_samples[i] * (l - baseline)
        
#         alpha_w_grad /= self.train_sample_loops
        
#         full_alpha_wgrad = torch.zeros_like(r_alpha_b, requires_grad=False)
#         full_alpha_wgrad[r2q_matches[:, 0]] = alpha_w_grad.to(r_alpha_b.device)
        
#         return baseline, full_alpha_wgrad, outlier_loss, sel_inlier_ratios
