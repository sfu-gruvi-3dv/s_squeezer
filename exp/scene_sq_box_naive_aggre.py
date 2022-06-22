from core_dl.lightning_logger import LightningLogger
from core_dl.lightning_model import BaseLightningModule
from core_dl.train_params import TrainParameters
import exp.scene_sq_visualizer as sq_vis
from exp.scene_sq_utils import *
from net.scene_fuser_sq import *
from dataset.common.base_data_source import Pt2dObs, Pt3dObs, ClipMeta
from dataset.common.split_scene import sel_subset_clip, sel_subset_obs2d
from net.loss import distinctive_loss


class SceneSQBox(BaseLightningModule):

    def __init__(self, params: TrainParameters) -> None:
        assert len(params.DEV_IDS) >= 2
        super().__init__(params)

        # parameters
        self.reg_loss_w = from_meta(self.args.AUX_CFG_DICT, 'reg_loss_w', default=0.5)

    def _set_network(self, args):
        """ configure the network
        """
        self.q2r = SuperGlueMatcher(device=self.args.DEV_IDS[0])
        self.sqz = SceneSqueezerWithTestQueries(args=args, kypt_matcher=self.q2r)

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

    def squeeze_scene_pts(self, vr_in: Tuple, anchor_in: Tuple, train_sqzd=True):
        vr_metas, vr_pt2d = vr_in
        ar_metas, ar_pt2d, ar_pt3d = anchor_in

        # compute distinctiveness score
        if not train_sqzd:
            with torch.no_grad():
                log_var, (r_xyz, r_feats) = self.sqz.forward(vr_metas, vr_pt2d, ar_metas, ar_pt2d, ar_pt3d)
        else:
            log_var, (r_xyz, r_feats) = self.sqz.forward(vr_metas, vr_pt2d, ar_metas, ar_pt2d, ar_pt3d)
            
        return log_var, r_xyz, r_feats

    """ Logs -----------------------------------------------------------------------------------------------------------
    """
    def log_visualization(self, batch_input, res_dict, vis_q2r=False):
        if self.logger is None:
            return

        q_imgs, _, q_info, rp_imgs, rp_info = batch_input[:5]

        # visualize distinctive score
        d_scores = asnumpy(self.sqz.logvar2score(res_dict['log_var']))
        vis_dist_s = sq_vis.plot_sel_ref_pts(rp_imgs, rp_info, d_scores, args={'pt_radius': 5})
        LightningLogger.add_image(self.logger.experiment, vis_dist_s, name='dist_pt', step=self.global_step)

        # visualize query to reference matches
        if vis_q2r:
            num_q_imgs = q_imgs.shape[1]
            vis_q2r = sq_vis.plot_q2r(sample=[q_imgs, q_info, rp_imgs, rp_info], res=res_dict, args={
                'sel_q_ids': [0, num_q_imgs // 2, num_q_imgs - 1],
                'max_ref_frames': 4,
                'num_corres': 10
            })
            LightningLogger.add_image(self.logger.experiment, vis_q2r, name='q2r_matches', step=self.global_step)

    """ Training routines ----------------------------------------------------------------------------------------------
    """
    def forward(self, input):
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
        res_dict = dict()
        with torch.cuda.device(self.dev_ids[0]):
            log_var, r_xyz, r_feats = self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d), 
                                                             anchor_in=(r_metas, r_pt2d, r_pt3d))
            if log_var is None:
                return None, None, None
            res_dict['log_var'] = log_var
            
        # step 2: query to scene registering
        with torch.cuda.device(self.dev_ids[0]) as _, torch.no_grad() as _:
            q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)
            q2r_scores = register_multi_q2r(self.q2r, q_sp_feats, r_feats)
            q2r_pred_matches = [self.q2r.get_matches(s, optimal_transport=True) for s in q2r_scores]

            for q_i, (score, pred_match) in enumerate(zip(q2r_scores, q2r_pred_matches)):
                res_dict[q_i] = {'P': score, 'matches': extract_matches(pred_match)}

        # step 3: compute loss
        with torch.cuda.device(self.dev_ids[0]):
            
            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
            pred_r2q = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})
            pred_r2q_dist = r2q_reproj_dist(q_metas, q_pt2d, r_pt3d, pred_r2q)

            # distinctive loss
            d_total_loss, d_loss1, d_loss2 = distinctive_loss(pred_r2q_dist, log_var, reg_loss_w=self.reg_loss_w)

        return {'total_loss': d_total_loss.mean(), 
                'p_loss': d_loss1.mean(), 
                'reg_loss': d_loss2.mean()}, res_dict, q_idx

    def training_step(self, batch, batch_idx):
        loss, res_dict, _ = self.forward(batch)
        if loss is None:
            return torch.zeros(1, requires_grad=True)

        # log
        self.log_dict(loss, prog_bar=True)

        if self.on_visualize():
            self.log_visualization(batch, res_dict)

        if self.global_step % 20 == 0 and self.logger is not None:  # histogram
            d_scores = asnumpy(self.sqz.logvar2score(res_dict['log_var']))
            LightningLogger.add_hist(self.logger.experiment, d_scores, name='d_scores', step=self.global_step)

        return loss['total_loss']

    """ Evaluation routines --------------------------------------------------------------------------------------------
    """
    # def validation_step(self, batch, batch_idx):
    #     loss, res_dict, _ = self.forward(batch)
    #
    #     return None
    #
    # def pred(input):
    #     pass