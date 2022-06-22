from core_io.serialize import dump_pickle
from typing import List, Tuple
from numpy.lib.arraysetops import isin
import torch
from torch import tensor
from core_io.meta_io import *
from torch_scatter import scatter
from einops import rearrange, repeat, asnumpy
from net.pt_transformer import *
import torch.nn as nn
import numpy as np
from matcher.superglue_matcher import BaseMatcher, SuperGlueMatcher
from dataset.common.hloc_db import Pt2dObs, Pt3dObs
from SuperGluePretrainedNetwork.models.superglue import normalize_keypoints
import torch.nn.functional as F
import time
import core_3dv.camera_operator_gpu as cam_opt_gpu
from core_dl.torch_ext import batch_sel_3d
from dataset.common.base_data_source import ClipMeta, Pt2dObs, Pt3dObs


def extract_matches(res: dict):
    m_q_idx = torch.nonzero(res['matches0'] > 0)[:, 1].cpu()
    m_ref_idx = res['matches0'].cpu()[0, m_q_idx]
    matches = torch.cat([m_q_idx.view(1, -1), m_ref_idx.view(1, -1)], dim=0).T
    return matches


def register_q_frame(q2r: SuperGlueMatcher, q: dict, ref: dict):
    """ register single query frames to the scene
    """
    cur_dev = torch.cuda.current_device()

    q_pt2d_npos_t = rearrange(normalize_2d_pos(q['dim_hw'], q['pos']), '(b n) v -> b n v', b=1)
    q_pt2d_feats_t = q2r.encode_pt_feats([rearrange(q['feats'], '(b n) c -> b c n', b=1),
                                          q_pt2d_npos_t,
                                          rearrange(q['scores'], '(b n) -> b n', b=1)])

    assert ref['feats'].shape[1] == q_pt2d_feats_t.shape[1]     # make sure ref dim: (B, C, N)
    res = q2r.forward({'desc': q_pt2d_feats_t.to(cur_dev)}, {'desc': ref['feats'].to(cur_dev)})

    # score matrix
    p_mat = res['scores']
    matches = extract_matches(res)

    return p_mat, matches



def register_q_frames(q2r: SuperGlueMatcher, q_meta:ClipMeta, q_pt2d: Pt2dObs, ref_feats: torch.Tensor):
    """ register multiple query frames to the scene
    """
    cur_dev = torch.cuda.current_device()
    num_q_frames = q_meta.num_frames()
    q_pt2d = q_pt2d.to_tensor(cur_dev)

    res_dict = dict()
    for q_id in range(num_q_frames):

        q_input = {'pos': q_pt2d.uv[q_id],
                   'feats': q_pt2d.feats[q_id],
                   'scores': q_pt2d.score[q_id],
                   'dim_hw': q_meta.dims[q_id]}
        p_mat, matches = register_q_frame(q2r, q=q_input, ref={'feats': ref_feats.to(cur_dev)})
        res_dict[q_id] = {'P': p_mat, 'matches': matches}

    return res_dict


def r2q_reproj_dist_pairwise(q_meta: ClipMeta, q_pos2d: Pt2dObs, r_pt3d: Pt3dObs):
    """
    pairwise reproj distance irrespective of matches
    """
    q_Tcws = [t.view(3, 4) for t in q_meta.Tcws]
    q_Ks = [q.view(3, 3) for q in q_meta.K]
    q_kypt_pos = [q.view(-1, 2) for q in q_pos2d.uv]
    r_3d_pts = r_pt3d.xyz.view(-1, 3)

    dists= []
    for q_idx, (q_K, q_Tcw) in enumerate(zip(q_Ks, q_Tcws)):
        rpj_3d_local = cam_opt_gpu.transpose(q_Tcw[:3, :3], q_Tcw[:3, 3], r_3d_pts)
        rpj_2d_pos, _ = cam_opt_gpu.pi(q_K, rpj_3d_local)

        dist = torch.cdist(
            rearrange(q_kypt_pos[q_idx], 'n c -> () n c'),
            rearrange(rpj_2d_pos, 'n c -> () n c'),
            p=2.0,
            compute_mode='donot_use_mm_for_euclid_dist'
        )
        dists.append(dist)

    return dists


class ObsFusion(nn.Module):

    def __init__(self, in_c=256, num_obs=2, beta_init=0.5, args=None):
        super(ObsFusion, self).__init__()

        self.sel_by_rpj_dist = from_meta(args, 'sel_by_rpj_dist', default=False)
        self.sel_by_rpj_topk = from_meta(args, 'sel_by_rpj_topk', default=50)

        self.appear_t = nn.Parameter(torch.ones(1) * beta_init)
        self.rpj_t = nn.Parameter(torch.ones(1) * beta_init)
        self.ln = nn.Conv1d(in_c * num_obs, in_c, kernel_size=1)

    def forward(self, appear_o2r_score, reproj_o2r_score, o_desc):
        """
        appear_o2r_score: appearnce similarity, dim: (B, TQ, R)
        reproj_o2r_score: reprojection similarity, dim: (B, TQ, R)
        o_desc: the observation feature, dim: (B, F, TQ)
        """
        cur_dev = torch.cuda.current_device()

        appear_o2r_s = appear_o2r_score.to(cur_dev)
        rpj_o2r_s = torch.exp(-(reproj_o2r_score + 1.2e-5).log()).to(cur_dev)     # dim: (B, TQ, R)

        if self.sel_by_rpj_dist:
            # select top_k
            _, topk_idx = torch.topk(rpj_o2r_s, self.sel_by_rpj_topk, dim=1) # dim: (B, K, R)

            appear_o2r_s = batch_sel_3d(appear_o2r_s, dim=1, index=topk_idx)
            rpj_o2r_s = batch_sel_3d(rpj_o2r_s, dim=1, index=topk_idx)

        appear_o2r_s = F.softmax(self.appear_t * appear_o2r_s, dim=1)
        rpj_o2r_s = F.softmax(self.rpj_t * rpj_o2r_s, dim=1)

        o_desc = o_desc.to(cur_dev)
        appear_tq2r_f = torch.einsum('bfq,bqr->bfr', o_desc, appear_o2r_s)
        rpj_tq2r_f = torch.einsum('bfq,bqr->bfr', o_desc, rpj_o2r_s)
        merged = self.ln(torch.cat([appear_tq2r_f, rpj_tq2r_f], dim=1))
        return merged


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


class Anchor2TestsFuser(nn.Module):
    def __init__(self, in_c, qk_c, args:dict):
        super(Anchor2TestsFuser, self).__init__()
        self.args = args

        self.in_c = in_c
        num_heads = from_meta(self.args, 'num_anchor2test_heads', default=4)

        self.k_proj = nn.Conv1d(in_c, num_heads * qk_c, kernel_size=1)
        self.q_proj = nn.Conv1d(in_c, num_heads * qk_c, kernel_size=1)
        self.v_proj = nn.Conv1d(in_c, num_heads * in_c, kernel_size=1)

        self.final_fc = nn.Conv1d(num_heads * in_c, in_c, kernel_size=1)
        self.num_heads = num_heads

    @staticmethod
    def attention(query, key, value):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
        prob = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

    def forward(self, anchor_feats: torch.Tensor, test_feats: torch.Tensor):
        """
        :param anchor_feats: (B, 256, A)
        :param test_feats: (B, T, 256， A)

        where,
        B: batch size
        A: number of anchor features
        T: number of test images
        256: descriptor dimension

        """

        # Note: cannot handle batch size because the
        # number of anchor features (na) acts like batch size and
        # 256 (descriptor dimension) is the embedding dimension
        assert(anchor_feats.shape[0] == 1)
        assert(test_feats.shape[0] == 1)
        assert(anchor_feats.shape[0] == 1)

        _, T, E, A = test_feats.shape
        H = self.num_heads
        assert(E == self.in_c)

        q = self.q_proj(anchor_feats.view(1, E, A))                         # dim: (1, H*Qc, A)
        k = self.k_proj(rearrange(test_feats, 'B T E A -> (B T) E A'))      # dim: (3, H*Qc, A)
        v = self.v_proj(rearrange(test_feats, 'B T E A -> (B T) E A'))      # dim: (3, H*E , A)

        x, prob = self.attention(
            query=rearrange(q, 'B (Qc H) A -> A Qc H B', H=H),
            key=rearrange(k, 'T (Qc H) A -> A Qc H T', H=H),
            value=rearrange(v, 'T (C H) A -> A C H T', H=H)
        )

        x = rearrange(x, 'A C H T -> A (C H) T')
        x = self.final_fc(x).squeeze(-1)

        return {
            'attn_output': x,
            'attn_weight': prob
        }


from evaluator.trajectory_eval import rel_distance, rel_R_deg
def sel_nearest_anchor(q_tcw, anchor_tcws: list):
    """function for finding the nearest anchor frame."""
    assert(len(anchor_tcws))

    a_tcws = [asnumpy(t.cpu()) for t in anchor_tcws]
    q_tcw_n = asnumpy(q_tcw)

    q2r_rel_deg = np.asarray([rel_R_deg(q_tcw_n[:3, :3], t[:3, :3]) for t in a_tcws])
    q2r_dist = np.asarray([rel_distance(q_tcw_n, t) for t in a_tcws])

    # TODO: soft-code the angle threshold
    invalid_anchors = np.abs(q2r_rel_deg) > 45.0
    q2r_dist[invalid_anchors] = np.inf

    q2r_d_sorted = np.argsort(q2r_dist)

    return q2r_d_sorted[0]


def is_in_t(ref_2d_pts: torch.Tensor, ref_2d_depth: torch.Tensor, dim_hw):
    """ Check the point is in the image plane
    """
    x = torch.logical_and(ref_2d_pts[:, 0] > 0, ref_2d_pts[:, 0] < dim_hw[1])
    y = torch.logical_and(ref_2d_pts[:, 1] > 0, ref_2d_pts[:, 1] < dim_hw[0])
    z = torch.logical_and(x, y)
    return torch.logical_and(z, ref_2d_depth.view(-1) > 0)


def encode_sp_feats(matcher, meta: ClipMeta, pt2d: Pt2dObs):
    cur_dev = torch.cuda.current_device()
    meta, pt2d = meta.to_tensor(cur_dev), pt2d.to_tensor(cur_dev)
    N = meta.num_frames()

    # encode features
    pt2d_npos_t = torch.cat([q.squeeze(0) for q in normalize_kpts_pos(meta.dims, pt2d.uv)])             # (VR, 2)
    M = pt2d_npos_t.shape[0]

    pt2d_en_feats = matcher.encode_pt_feats([
        rearrange(torch.cat(pt2d.feats), 'm c -> () c m', m=M),
        rearrange(pt2d_npos_t, 'm v -> () m v', m=M, v=2),
        rearrange(torch.cat(pt2d.score).view(-1), 'm -> () m', m=M)
    ])
    pt2d_en_feats = torch.split(pt2d_en_feats, [i.shape[0] for i in pt2d.feats], dim=-1)

    return pt2d_en_feats


def register_multi_q2r(matcher, q_meta, vr_pt2d_feats, r_meta, r_pt3d_xyz, r_pt3d_feats, r_pt3d_score):
    cur_dev = torch.cuda.current_device()
    N = r_pt3d_xyz.shape[0]
    assert r_pt3d_feats.shape[0] == N

    v2r_scores = []

    for vr_id, vr_desc in enumerate(vr_pt2d_feats):
        # step 1: find the nearest anchor
        nearest_anchor_id = sel_nearest_anchor(q_meta.Tcws[vr_id], r_meta.Tcws)
        nearest_anchor_Tcw = r_meta.Tcws[nearest_anchor_id]

        # step 2: re-project the 3d point to anchor frame
        r_local_xyz = cam_opt_gpu.transpose(nearest_anchor_Tcw[:3, :3], nearest_anchor_Tcw[:3, 3], r_pt3d_xyz)
        r_rpj_2duv, vr_rpj_depth = cam_opt_gpu.pi(r_meta.K[nearest_anchor_id], r_local_xyz)

        # step 3: generate mask
        r_valid_pts = is_in_t(r_rpj_2duv, vr_rpj_depth, r_meta.dims[nearest_anchor_id])  # (N, ) boolean array
        r_valid_idx = torch.where(r_valid_pts == True)[0]                   # select the points inside of image plane

        if r_valid_idx.shape[0] == 0:
            v2r_s = torch.zeros(1, vr_desc.shape[-1], r_pt3d_xyz.shape[0], device=vr_desc.device)
            v2r_scores.append(v2r_s)
            continue

        r_valid_n_pts = normalize_2d_pos(r_meta.dims[nearest_anchor_id], r_rpj_2duv[r_valid_pts])
        r_valid_feats = r_pt3d_feats[r_valid_idx, :]             # (M, C)
        r_valid_scores = r_pt3d_score[r_valid_idx, :].view(-1)   # (M, )
        M, C = r_valid_feats.shape[:2]

        r_valid_en_feats = matcher.encode_pt_feats([
            rearrange(r_valid_feats, 'm c -> () c m', m=M).to(cur_dev),
            rearrange(r_valid_n_pts, 'm uv -> () m uv', m=M).to(cur_dev),
            rearrange(r_valid_scores, 'm -> () m').to(cur_dev)
        ])                                                                               # (B, C, N)

        # step 4: run super-glue
        v2rv_s = matcher.get_score(
            vr_desc,
            r_valid_en_feats,
            optimal_transport=False
        )

        # step 5: padding the zeros to invalid columns
        v2r_s = torch.zeros(1, vr_desc.shape[-1], r_pt3d_xyz.shape[0], device=v2rv_s.device)
        v2r_s[:, :, r_valid_idx] = v2rv_s
        v2r_scores.append(v2r_s)

    return v2r_scores


class SceneSqueezerWithTestQueries(nn.Module):

    def __init__(self, args:dict, kypt_matcher: SuperGlueMatcher):
        super(SceneSqueezerWithTestQueries, self).__init__()
        self.args = args
        self.matcher = kypt_matcher
        self.pt_transformer = BasePointTransformer(dim=[512, 256, 256, 256, 512])

        self.obs_fuser = ObsFusion()
        self.anchor2test_fuser = Anchor2TestsFuser(in_c=256, qk_c=64, args=self.args)

        # parameters
        self.move_to_origin = from_meta(self.args, 'move_to_origin', True)
        self.aggre_method = from_meta(self.args, 'sqz_aggre_method', default='mean')

    def register_q_frames(self, q: dict, ref: dict):
        return register_q_frames(self.matcher, q, ref)

    def forward(self, vr_meta: ClipMeta, vr_pt2d: Pt2dObs, r_meta: ClipMeta, r_pt2d: Pt2dObs, r_pt3d: Pt3dObs):
        cur_dev = torch.cuda.current_device()

        # pre-process verifications
        with torch.no_grad():
            vr_pt2d_feats = encode_sp_feats(self.matcher, vr_meta, vr_pt2d)                         # [(num_VR, C)]

        # references points
        with torch.no_grad():
            r_meta = r_meta.to_tensor(cur_dev)
            r_pt2d = r_pt2d.to_tensor(cur_dev)
            r_pt3d = r_pt3d.to_tensor(cur_dev)

            # aggregate 2D features for each 3D point.
            r_pt3d_ids = torch.cat(r_pt2d.obs3d_ids)
            r_sp_feats = torch.cat(r_pt2d.feats)
            r_aggr_feats = aggr_pc_feats(r_sp_feats, r_pt3d_ids, aggr_method=self.aggre_method)
            r_aggr_scores = aggr_pc_feats(torch.cat(r_pt2d.score).unsqueeze(-1), r_pt3d_ids,
                                          aggr_method=self.aggre_method)

        pt3d_xyz = r_pt3d.xyz
        if self.move_to_origin:
            pt3d_xyz = move_to_origin(pt3d_xyz.clone())
            pt3d_xyz = normalize_pts(pt3d_xyz)

        # Observation from query to references.
        with torch.no_grad():
            reproj_dists = r2q_reproj_dist_pairwise(vr_meta, vr_pt2d, r_pt3d)
            v2r_sglue_obs = register_multi_q2r(self.matcher, vr_meta, vr_pt2d_feats,
                                               r_meta, r_pt3d.xyz, r_aggr_feats, r_aggr_scores)

            v2r_obs = [(v2r_s, reproj_dist) for v2r_s, reproj_dist in zip(v2r_sglue_obs, reproj_dists)]

        # fuse the observation individually
        v2r_obs_fused = []
        for vr_id, (v2r_s, reproj_dist) in enumerate(v2r_obs):
            vr_desc = vr_pt2d_feats[vr_id]
            t = self.obs_fuser.forward(v2r_s, reproj_dist.detach(), vr_desc)  # dim: (1, 256, R)
            v2r_obs_fused.append(t.view(1, 256, -1))
        v2r_obs_fused = torch.cat(v2r_obs_fused, dim=0)

        # get distinctive scores by forwarding with point transformer
        r_aggr_feats = rearrange(r_aggr_feats, 'r c -> () c r').to(cur_dev)
        v2r_obs_fused = rearrange(v2r_obs_fused, 't c r -> () t c r').to(cur_dev)
        aggre_obs_feats = self.anchor2test_fuser(r_aggr_feats, v2r_obs_fused)

        in_feats = aggre_obs_feats['attn_output']                                           # (N, C)
        if torch.sum(torch.isnan(in_feats)).item() > 0:
            print('A')
            return None, (None, None, None)

        # point-transformer to fuse all points
        in_feats = rearrange(in_feats, 'n c -> () n c')
        r_aggr_feats = rearrange(r_aggr_feats, 'b c n -> b n c', b=1).to(cur_dev)
        in_feats = torch.cat([in_feats, r_aggr_feats], dim=-1)
        _, N, _ = in_feats.shape

        log_var = self.pt_transformer.forward((pt3d_xyz.view(1, N, 3), in_feats)).view(1, N)

        return log_var, (r_pt3d.xyz, r_aggr_feats.squeeze(0), r_aggr_scores)
