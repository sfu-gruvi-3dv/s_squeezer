from exp.scene_sq_unified_ptsel_box import *
import math
# import ipdb

def measure_r2q_inlier_ratio(q_K, q_Tcw, q_pos2d, r_xyz, r2q_matches_valid, rpj_thres=12):
    rpjq_2d_pos, _ = cam_opt_gpu.reproject(q_Tcw, q_K, r_xyz.to(q_Tcw.device))
    r_q_sel_pos, q_sel_pos = corres_pos_from_pairs(rpjq_2d_pos, q_pos2d, r2q_matches_valid.to(q_pos2d.device))
    r2q_rpj_err = torch.norm(r_q_sel_pos - q_sel_pos, dim=1).cpu()
    ratio = torch.sum(r2q_rpj_err < rpj_thres) / r2q_rpj_err.shape[0]
    return ratio


def forward_dual_matcher(self, input: list, return_type='train'):

    # freeze_bn_layer(self.q2r)
    self.q2r.eval()
    self.sqz.eval()
    # freeze_bn_layer(self.dual_matcher)

    fq_imgs, fq_segs, fq_info, rp_imgs, rp_info = input[:5]

    # parse the dict
    fq_metas, fq_pt2d, _ = dict2obs(fq_info)
    r_metas, r_pt2d, r_pt3d = dict2obs(rp_info)

    if r_pt3d.num_pts() < self.pt_sel.qp_num_pts or fq_metas.num_frames() < 2:
        return None, None, None

    # split the verification, query set
    q_idx, vr_idx = asnumpy(fq_info['q_idx']), asnumpy(fq_info['vr_idx'])
    q_metas, q_pt2d = sel_subset_clip(fq_metas, q_idx), sel_subset_obs2d(fq_pt2d, q_idx)
    vr_metas, vr_pt2d = sel_subset_clip(fq_metas, vr_idx), sel_subset_obs2d(fq_pt2d, vr_idx)

    # step 1: build scene representation and squeeze the points ----------------------------------------------------
    res_dict = dict()
    with torch.cuda.device(self.device_of(self.sqz)) as _, torch.no_grad() as _:
        log_var, r_kernel, r_xyz, r_feats = \
            self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d), anchor_in=(r_metas, r_pt2d, r_pt3d), learnt_kernel=False)
        if log_var is None:
            return None, None, None
        log_var = log_var.detach()

    # step 2: selected by qp ---------------------------------------------------------------------------------------
    with torch.cuda.device(self.device_of(self.pt_sel)) as _, torch.no_grad() as _:
        cur_dev = torch.cuda.current_device()
        dist_score = torch.exp(- log_var).view(-1)
        dist_score, r_kernel = dist_score.to(cur_dev), r_kernel.to(cur_dev)

        if self.qp_sel_random:
            sel_pts_idx = np.arange(0, dist_score.shape[0])
            np.random.shuffle(sel_pts_idx)
        else:
            sel_pts_idx = asnumpy(torch.argsort(dist_score.view(-1), descending=True))

        sel_pts_idx = sel_pts_idx[:self.pt_sel.qp_num_pts]
        alpha = self.pt_sel.sel_by_qp(r_kernel, dist_score, sel_idx=sel_pts_idx)
        if alpha is None:
            return None, None, None
        alpha /= self.pt_sel.get_pt_sel_thres()

        # filtering
        sel_ref_idx = torch.where(alpha > 0.3)[0]
        r_xyz = r_xyz[sel_ref_idx]
        r_feats = r_feats[:, :, sel_ref_idx]
        # ipdb.set_trace()
        # change the dimension of features
        if self.is_reduce_dim:
            r_feats = self.reduce_dim(r_feats)
            r_feats = self.sigmoid(r_feats) - 0.5
            r_feats, quanti_r_feats = self.quantizate(r_feats)
            r_feats = self.increase_dim(r_feats)

    # step 3: match the selected reference points ------------------------------------------------------------------
    with torch.cuda.device(self.device_of(self.q2r)) as _, torch.no_grad() as _:
        q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)

    with torch.cuda.device(self.device_of(self.dual_matcher)) as _:
        cur_dev = torch.cuda.current_device()
        q_sp_feats = [q_f.to(cur_dev) for q_f in q_sp_feats]
        r_feats = r_feats.to(cur_dev)

        # gt matches -----------------------------------------------------------------------------------------------
        gt_r2qs = []
        for q_i in range(q_metas.num_frames()):
            gt_q2r = gen_gt_corres(q_pt2d.uv[q_i], q_metas.K[q_i], q_metas.Tcws[q_i],
                                   None, q_metas.dims[q_i], r_xyz, reproj_dist_thres=8.0, only_valid_pairs=True)
            if gt_q2r is None:
                gt_r2qs.append(None)
                continue
            gt_r2q = inverse_matches(gt_q2r.cpu())
            full_gt_r2q = torch.cat([torch.arange(0, r_xyz.shape[0]).unsqueeze(-1).int(),
                                     -1 * torch.ones(r_xyz.shape[0]).unsqueeze(-1).int()], dim=-1)
            full_gt_r2q[gt_r2q[:, 0], 1] = gt_r2q[:, 1].view(-1).int()
            gt_r2qs.append(full_gt_r2q)
        
        if self.training:
            optimizer = self.optimizers()
            optimizer.zero_grad()

        if self.args.DEBUG:
            r2q_pred_matches, r2q_losses, refined_gt_r2qs = [], [], []

        r2q_avg_inliers_ratios, r2q_visible_losses, r2q_invisible_losses = [], [], []
        total_loss = 0
        
        for q_i in range(min(q_metas.num_frames(), 3)):
            if gt_r2qs[q_i] == None:
                continue
            
            r2q_score = self.dual_matcher.get_score(r_feats, q_sp_feats[q_i], optimal_transport=True)
            r2q_matches = self.dual_matcher.get_matches(r2q_score, optimal_transport=False)['matches0'][0].cpu()
            r2q_matches = torch.cat([torch.arange(0, r_xyz.shape[0]).int().unsqueeze(-1),
                                     r2q_matches.int().unsqueeze(-1)], dim=-1).long()
            r2q_gt_matches = gt_r2qs[q_i].long()

            # reproj error: ref to query
            repj_r_pos2d, repj_r_depth = cam_opt_gpu.reproject(q_metas.Tcws[q_i], q_metas.K[q_i], r_xyz)
            repj_r2q_visible = cam_opt_gpu.is_in_t(repj_r_pos2d, repj_r_depth, dim_hw=[1024, 1024])
            r2q_matches_v_idx = torch.where(r2q_matches[:, 1] > 0)[0]
            r2q_matches_valids = r2q_matches[r2q_matches_v_idx, :].long()

            repj_sel_pos2d, q_sel_pos2d = corres_pos_from_pairs(repj_r_pos2d, q_pt2d.uv[q_i], r2q_matches_valids)
            r2q_matches_valids_rpj_err = torch.ones(r_xyz.shape[0]) * 50
            r2q_matches_valids_rpj_err[r2q_matches_v_idx] = torch.norm(repj_sel_pos2d - q_sel_pos2d, dim=1).cpu()

            # replace the `r2q_gt_matches` if repj_err is low but r2q_matches != r2q_gt_matches

            r2q_gt_matches[:, 1] = torch.where(
                torch.logical_and(r2q_matches_valids_rpj_err < 12.0, r2q_gt_matches[:, 1] != r2q_matches[:, 1]), 
                r2q_matches[:, 1], r2q_gt_matches[:, 1])

            # loss for visible matches
            r2q_score_exp = r2q_score[0].exp().contiguous()
            r2q_gt_matches_v = r2q_gt_matches[torch.where(r2q_gt_matches[:, 1] > 0)[0]].long()
            r2q_loss = exp_loss(r2q_gt_matches_v.to(r2q_score.device), r2q_score_exp)

            # loss for in-visible matches
            # check if invisible but predicted
            r2q_gt_matches_iv_ids = torch.where(r2q_gt_matches[:, 1] < 0)[0]
            r2q_matches_iv = torch.where(r2q_matches[r2q_gt_matches_iv_ids, 1] > 0)[0]
            r2q_score_iv_dustbin = r2q_score_exp[:, -1][r2q_matches_iv]
            r2q_inv_loss = -torch.log(r2q_score_iv_dustbin + 1e-6)

            # gather all losses
            r2q_mloss, r2q_inv_mloss = r2q_loss.mean(), r2q_inv_loss.mean()
            if r2q_inv_loss.shape[0] == 0:
                r2q_inv_mloss = torch.tensor(0.0, requires_grad=True).to(r2q_inv_loss.device)
            else:
                r2q_invisible_losses.append(r2q_inv_mloss.item())

            if r2q_loss.shape[0] == 0:
                r2q_mloss = torch.tensor(0.0, requires_grad=True).to(r2q_loss.device)
            else:
                r2q_visible_losses.append(r2q_mloss.item())

            total_loss = r2q_mloss + 0.1 * r2q_inv_mloss

            # measure the inlier ratios
            inlier_ratio = \
                measure_r2q_inlier_ratio(q_metas.K[q_i], q_metas.Tcws[q_i], q_pt2d.uv[q_i], r_xyz, r2q_matches_valids)
            
            if not math.isnan(inlier_ratio):
                r2q_avg_inliers_ratios.append(inlier_ratio)

            if self.training and total_loss > 0:
                total_loss.backward()

            if self.args.DEBUG:
                refined_gt_r2qs.append(r2q_gt_matches_v)
                r2q_pred_matches.append(r2q_matches_valids)
                r2q_losses.append(r2q_loss)

        if self.training:
            optimizer.step()

        if return_type == 'debug':
            if len(r2q_avg_inliers_ratios) == 0:
                inlier_ratios_avg = torch.tensor(0., dtype=float)
            else:
                inlier_ratios_avg = sum(r2q_avg_inliers_ratios) / len(r2q_avg_inliers_ratios)
            return q_metas, q_pt2d, refined_gt_r2qs, r2q_pred_matches, r2q_losses, r_xyz, inlier_ratios_avg
        
        if len(r2q_invisible_losses) == 0:
            r2q_invisible_losses.append(0)
        if len(r2q_visible_losses) == 0:
            r2q_visible_losses.append(0)
        if len(r2q_avg_inliers_ratios) == 0:
            r2q_avg_inliers_ratios.append(0)
        
    return {"dual_vis_mloss": sum(r2q_visible_losses) / len(r2q_visible_losses),
            "dual_invis_mloss": sum(r2q_invisible_losses) / len(r2q_invisible_losses),
            "r2q_avg_inliers": sum(r2q_avg_inliers_ratios) / len(r2q_avg_inliers_ratios),
            "total_loss": total_loss}, None, None
