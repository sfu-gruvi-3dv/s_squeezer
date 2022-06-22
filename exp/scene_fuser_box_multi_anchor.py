# from _typeshed import Self
# from math import dist
import torch

from core_dl.lightning_logger import LightningLogger
from core_dl.lightning_model import BaseLightningModule
from core_dl.train_params import TrainParameters
from dataset.common.gt_corres_torch import *
from exp.scene_sq_utils import *
from net.scene_fuser_multi_anchor import *
from dataset.common.base_data_source import Pt2dObs, Pt3dObs, ClipMeta
from dataset.common.split_scene import sel_subset_clip, sel_subset_obs2d

def dict2obs(info: dict):
    clip = ClipMeta.from_dict(info, to_numpy=False)
    pt2d_obs = Pt2dObs.from_dict(info, to_numpy=False)
    if 'pt3d' in info:
        pt3d_obs = Pt3dObs.from_dict(info, to_numpy=False)
    else:
        pt3d_obs = None
    return clip, pt2d_obs, pt3d_obs


class SceneSQBox(BaseLightningModule):

    def __init__(self, params: TrainParameters) -> None:
        assert len(params.DEV_IDS) >= 2
        super().__init__(params)

    def _set_network(self, args):
        """ configure the network
        """
        self.diff_b_k = from_meta(self.args.AUX_CFG_DICT, 'net_diff_b_k', default=50)
        self.q2r = SuperGlueMatcher(device=self.args.DEV_IDS[0])
        self.sqz = SceneSqueezerWithTestQueries(args=args, kypt_matcher=self.q2r).to(self.args.DEV_IDS[0])

        self.reg_loss_w = from_meta(self.args.AUX_CFG_DICT, 'reg_loss_w', default=0.5)

    def _instance_scripts(self):
        return [self, self.q2r, self.sqz, './train.sh']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.START_LR)
        return optimizer

    def __move_to_devices__(self, dev_ids):
        self.q2r = self.q2r.to(dev_ids[0])
        self.sqz = self.sqz.to(dev_ids[0])

    """ Pipeline -------------------------------------------------------------------------------------------------------
    """
    def register_q_frames(self, q_meta: ClipMeta, q_pt2d: Pt2dObs, ref_feats: torch.Tensor):
        return register_q_frames(self.q2r, q_meta, q_pt2d, ref_feats)

    def squeeze_scene_pts(self, vr_in: Tuple, anchor_in: Tuple):
        vr_metas, vr_pt2d = vr_in
        ar_metas, ar_pt2d, ar_pt3d = anchor_in

        log_var, (r_xyz, r_feats, r_scores) = self.sqz.forward(vr_metas, vr_pt2d, ar_metas, ar_pt2d, ar_pt3d)

        return log_var, r_xyz, r_feats, r_scores

    """ Training routines ----------------------------------------------------------------------------------------------
    """

    def gen_gt_matches(self, q_info, q_idx, ref_pt3ds, q_segs=None, reproj_thres=5):
        q_2d_pts = q_info['pt2d_pos']
        q_Ks = q_info['K']
        q_Tcws = q_info['Tcws']
        q_dims = [(int(dim[0].item()), int(dim[1].item())) for dim in q_info['dims']]
        q_mask = None if q_segs is None else mask_by_seg(q_segs, exclude_seg_label=[20, 80])[0]

        q2r_gt = gen_gt_corres(q_2d_pts[q_idx][0], q_Ks[q_idx][0], q_Tcws[q_idx][0], q_mask[q_idx],
                               q_dim_hw=q_dims[q_idx], ref_3d_pts=ref_pt3ds[0], reproj_dist_thres=reproj_thres)
        return q2r_gt

    def forward(self, input, q_id=None):
        q_imgs, q_segs, q_info, rp_imgs, rp_info = input[:5]

        # parse the dict
        fq_metas, fq_pt2d, _ = dict2obs(q_info)
        r_metas, r_pt2d, r_pt3d = dict2obs(rp_info)
        if r_pt3d.num_pts() < 128 or fq_metas.num_frames() < 2:
            return None, None, None

        # split the verification, query set
        q_idx, vr_idx = asnumpy(q_info['q_idx']), asnumpy(q_info['vr_idx'])
        q_metas, q_pt2d = sel_subset_clip(fq_metas, q_idx), sel_subset_obs2d(fq_pt2d, q_idx)
        vr_metas, vr_pt2d = sel_subset_clip(fq_metas, vr_idx), sel_subset_obs2d(fq_pt2d, vr_idx)

        # step 1: build scene representation and squeeze the points
        with torch.cuda.device(self.dev_ids[0]):
            log_var, r_xyz, r_feats, r_scores = self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d),
                                                                       anchor_in=(r_metas, r_pt2d, r_pt3d))
            if log_var is None:
                return None, None, None

        # step 2: query to scene registering
        with torch.cuda.device(self.dev_ids[0]) as _, torch.no_grad() as _:
            q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)
            q2r_scores = register_multi_q2r(self.q2r, q_metas, q_sp_feats, r_metas, r_xyz, r_feats, r_scores)
            q2r_pred_matches = [self.q2r.get_matches(s, optimal_transport=True) for s in q2r_scores]

            res_dict = dict()
            for q_i, (score, pred_match) in enumerate(zip(q2r_scores, q2r_pred_matches)):
                res_dict[q_i] = {
                    'P': score,
                    'matches': extract_matches(pred_match)
                }

        # step 3: compute loss
        with torch.cuda.device(self.dev_ids[0]):
            cur_dev = torch.cuda.current_device()

            ref_pt3d, num_ref_pts = r_pt3d.xyz, r_pt3d.num_pts()
            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]

            pred_r2q = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})
            pred_r2q_dist = r2q_reproj_dist(q_metas, q_pt2d, r_pt3d, pred_r2q)

            rpj_avg_dist, rpj_avg_var = torch.zeros(num_ref_pts), torch.zeros(num_ref_pts)
            for r in pred_r2q_dist.keys():
                rp = np.asarray(pred_r2q_dist[r]).ravel()
                rp_valids = rp < 6.0
                rpj_avg_dist[r] = 1 - (rp_valids.sum() / rp_valids.shape[0])

            # compute loss
            log_var = log_var.to(cur_dev)
            loss1 = torch.exp(-log_var) * (rpj_avg_dist.to(cur_dev))
            loss2 = self.reg_loss_w * log_var

            valid_r_idx = torch.Tensor([r for r in pred_r2q_dist.keys()]).long()
            loss1 = loss1.view(-1)[valid_r_idx]
            loss2 = loss2.view(-1)[valid_r_idx]
            total_loss = (loss1 + loss2).mean()

        res_dict['log_var'] = log_var
        res_dict['refs'] = (r_xyz, r_feats, r_scores)

        return {'total_loss': total_loss, 'p_loss': loss1.mean(), 'reg_loss': loss2.mean()}, res_dict, q_idx

    def training_step(self, batch, batch_idx):
        loss, res_dict, _ = self.forward(batch)
        if loss is None:
            return torch.zeros(1, requires_grad=True)

        self.log_dict({tag: l.item() for tag, l in loss.items()}, prog_bar=True)
        var = torch.sqrt(res_dict['log_var'].exp()).cpu()

        # log
        if self.global_step % self.args.VIS_STEPS == 0 and self.logger is not None:
            q_imgs, _, q_info, rp_imgs, rp_info = batch[:5]
            num_q_imgs = q_imgs.shape[1]

            # vis_q2r = sq_vis.plot_q2r(sample=[q_imgs, q_info, rp_imgs, rp_info], res=res_dict, args={
            #     # 'sel_q_ids': [0, num_q_imgs // 2, num_q_imgs - 1],
            #     'max_ref_frames': 4,
            #     'num_corres': 10
            # })

            # vis_sel_pt = sq_vis.plot_sel_ref_pts(rp_imgs, rp_info, res_dict['b'], args={'pt_radius': 5})
            # vis_dist_s = sq_vis.plot_sel_ref_pts(rp_imgs, rp_info, res_dict['d'], args={'pt_radius': 5})
            loggers = self.logger.experiment
            # LightningLogger.add_image(loggers, vis_q2r, name='q2r', step=self.global_step)
            # LightningLogger.add_image(loggers, vis_sel_pt, name='sel_pt', step=self.global_step)
            # LightningLogger.add_image(loggers, vis_dist_s, name='dist_pt', step=self.global_step)
            LightningLogger.add_hist(loggers, var, name='var', step=self.global_step)
            # LightningLogger.add_hist(loggers, res_dict['d'], name='D', step=self.global_step)

        elif self.global_step % 20 == 0 and self.logger is not None:

            loggers = self.logger.experiment
            LightningLogger.add_hist(loggers, var, name='var', step=self.global_step)
            # LightningLogger.add_hist(loggers, res_dict['d'], name='D', step=self.global_step)

        return loss['total_loss']
