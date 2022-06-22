import numpy as np
import torch
import torch.nn.functional as F
import core_3dv.camera_operator_gpu as cam_opt_gpu
from core_dl.torch_ext import index_of_elements
from einops import asnumpy, rearrange
from matcher.superglue_matcher import SuperGlueMatcher
from SuperGluePretrainedNetwork.models.superglue import normalize_keypoints
from dataset.common.base_data_source import ClipMeta, Pt2dObs, Pt3dObs
from dataset.common.gt_corres_torch import *

def dict2obs(info: dict):
    clip = ClipMeta.from_dict(info, to_numpy=False)
    pt2d_obs = Pt2dObs.from_dict(info, to_numpy=False)
    if 'pt3d' in info:
        pt3d_obs = Pt3dObs.from_dict(info, to_numpy=False)
    else:
        pt3d_obs = None
    return clip, pt2d_obs, pt3d_obs


def split_info(info: dict, indices):
    num_frames = len(info['img_names'])

    split_out = dict()
    for key, items in info.items():
        if isinstance(items, list) and len(items) == num_frames:
            split_out[key] = [items[i] for i in indices]
        else:
            split_out[key] = items

    return split_out


def r2q(q2r_matches: dict):
    r2q_matches = dict()
    for q, matches in q2r_matches.items():

        matches = q2r_matches[q]
        if isinstance(matches, torch.Tensor):
            matches = asnumpy(matches)

        q_ids, r_ids = matches[:, 0], matches[:, 1]

        for q_idx, r_idx in zip(q_ids, r_ids):
            if r_idx not in r2q_matches:
                r2q_matches[r_idx] = list()

            r2q_matches[r_idx].append((q, q_idx))
    return r2q_matches


def extract_matches_r2q(r2q_matches, q_idx):
    r2q_m = dict()
    for r, q_s in r2q_matches.items():
        q_s_d = {q_id: q_pt for (q_id, q_pt) in q_s}
        if q_idx in q_s_d:
            r2q_m[r] = q_s_d[q_idx]
    r2q_l = [(r, q) for r, q in r2q_m.items()]
    return np.asarray(r2q_l)


def r2q_reproj_dist(q_meta: ClipMeta, q_pt2d: Pt2dObs, r_pt3d: Pt3dObs, r2q_dict):
    q_Tcws = [t.view(3, 4).cpu() for t in q_meta.Tcws]
    q_Ks = [q.view(3, 3).cpu() for q in q_meta.K]
    q_kypt_pos = [q.view(-1, 2).cpu() for q in q_pt2d.uv]
    r_3d_pts = r_pt3d.xyz.view(-1, 3).cpu()

    rpj_2d_pts, rpj_2d_dpths = [], []
    for q_K, q_Tcw in zip(q_Ks, q_Tcws):
        rpj_3d_local = cam_opt_gpu.transpose(q_Tcw[:3, :3], q_Tcw[:3, 3], r_3d_pts)
        rpj_2d_pos, rpj_2d_dpth = cam_opt_gpu.pi(q_K, rpj_3d_local)
        rpj_2d_pts.append(rpj_2d_pos)
        rpj_2d_dpths.append(rpj_2d_dpth)

    r_rpj_dist = dict()
    for r, obs in r2q_dict.items():

        if r not in r_rpj_dist:
            r_rpj_dist[r] = list()

        for (q_f_idx, q_pt_idx) in obs:
            q_pt_pos = q_kypt_pos[q_f_idx][q_pt_idx].view(2)
            rpj_2d_pos = rpj_2d_pts[q_f_idx][r].view(2)
            dist = torch.norm(q_pt_pos - rpj_2d_pos)
            r_rpj_dist[r].append((q_f_idx, dist.item()))

    return r_rpj_dist


def r2q_scores(r2q_dict, q2r_score_dict):

    r2q_scores = dict()
    for r, obs in r2q_dict.items():

        if r not in r2q_scores:
            r2q_scores[r] = list()

        for (q_f_idx, q_pt_idx) in obs:
            score = q2r_score_dict[q_f_idx][q_pt_idx, r]
            r2q_scores[r].append((q_f_idx, score.item()))

    return r2q_scores


def r_conf(r_pt2d: Pt2dObs):
    r_scores = dict()
    for obs, scores in zip(r_pt2d.obs3d_ids, r_pt2d.score):
        obs = asnumpy(obs.view(-1))
        scores = asnumpy(scores.view(-1))

        for i, r in enumerate(obs):
            if r not in r_scores:
                r_scores[r] = []
            r_scores[r].append(scores[i])
    return r_scores


def gen_gt_matches(self, q_info, q_idx, ref_pt3ds, q_segs=None, reproj_thres=5):
    q_2d_pts = q_info['pt2d_pos']
    q_Ks = q_info['K']
    q_Tcws = q_info['Tcws']
    q_dims = [(int(dim[0].item()), int(dim[1].item())) for dim in q_info['dims']]
    q_mask = None if q_segs is None else mask_by_seg(q_segs, exclude_seg_label=[20, 80])[0]

    q2r_gt = gen_gt_corres(q_2d_pts[q_idx][0], q_Ks[q_idx][0], q_Tcws[q_idx][0], q_mask[q_idx],
                           q_dim_hw=q_dims[q_idx], ref_3d_pts=ref_pt3ds[0], reproj_dist_thres=reproj_thres)
    return q2r_gt


def normalize_kpts_pos(img_shapes, kpts_pos):
    """
        Normalize the keypoint 2d position from -1 to 1
    Args:
        img_shapes (tuple or list): image dimensions
        kpts_pos (Tensor): the keypoint 2d positions (u, v)

    Returns:
        normalized keypoint position.

    """
    img_dims = [(1, 3, int(shape[0]), int(shape[1])) for shape in img_shapes]
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
    return pt3d_xyz - pt3d_xyz.mean(dim=0).view(1, 3)


def normalize_3dpts(pt3d_xyz):
    scale_x = pt3d_xyz[:, 0].max() - pt3d_xyz[:, 0].min()
    scale_y = pt3d_xyz[:, 1].max() - pt3d_xyz[:, 1].min()
    scale_z = pt3d_xyz[:, 2].max() - pt3d_xyz[:, 2].min()
    max_dim = max(max(scale_x, scale_y), scale_z)
    pt3d_xyz /= (max_dim + 1e-5)
    return pt3d_xyz * 2


def get_corres_ref_2d_indices(
        matches: torch.Tensor, r_obs3d: torch.Tensor
) -> torch.Tensor:
    corres_q_inds = matches[:, 0]
    corres_r_inds = matches[:, 1]

    sel_r_idx = index_of_elements(r_obs3d, corres_r_inds)
    valid_rows = torch.where(sel_r_idx != -1)[0]
    sel_matches = torch.cat(
        [corres_q_inds[valid_rows].view(-1, 1), sel_r_idx[valid_rows].view(-1, 1)], dim=1
    ).to(r_obs3d.device)

    return sel_matches

