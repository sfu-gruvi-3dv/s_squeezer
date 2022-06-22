import numpy as np
import torch
import core_3dv.camera_operator_gpu as cam_opt_gpu
from net.scene_sq import corres_pos_from_pairs


def mark_corres(pred_matches, q_pos_2d, q_K, q_Tcw, ref_3d_pos, thres=5):

    ref_3d_pos = ref_3d_pos.view(-1, 3)
    q_pos_2d = q_pos_2d.view(-1, 2)
    q_Tcw = q_Tcw.view(3, 4)
    q_K = q_K.view(3, 3)

    ref_3d_lpos = cam_opt_gpu.transpose(q_Tcw[:3, :3], q_Tcw[:3, 3], ref_3d_pos)
    reprj_2d_pos, reprj_2d_depth = cam_opt_gpu.pi(q_K, ref_3d_lpos)

    q_corres_pos, reprj_corres_pos = corres_pos_from_pairs(q_pos_2d, reprj_2d_pos, pred_matches)
    dist = torch.norm(q_corres_pos - reprj_corres_pos, dim=1)
    flag = torch.logical_and((dist < thres).view(-1), (reprj_2d_depth[pred_matches[:, 1]] > 0).view(-1))

    return flag


def precision_q2r(flag, ):
    return flag.sum() / flag.shape[0]
