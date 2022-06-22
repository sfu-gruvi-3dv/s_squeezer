from core_dl.module_util import freeze_bn_layer
from itertools import chain
from numpy import bool_

import torch.cuda
import exp.scene_sq_visualizer as sq_vis
from core_dl.lightning_logger import LightningLogger
from core_dl.lightning_model import BaseLightningModule
from core_dl.train_params import TrainParameters
from dataset.common.split_scene import sel_subset_clip, sel_subset_obs2d
from net.dlt_pnp_loss import DLTPnPLoss
from net.fast_pnp_loss import FastPnPLoss
from net.qp_ptsel_transformer import PointSelection, compute_rbf_kernel
from net.scene_mfuser_sq import *
from core_dl.expr_ctx import ExprCtx
from net.loss import distinctive_loss, exp_loss
from matcher.superglue_gnn_matcher import SuperGlueGNNMatcher
from exp.scene_sq_dual_matcher import forward_dual_matcher
from exp.scene_sq_quat import forward_soft_quant
from net.soft_quant import SoftQuant


class SceneSQPTSelBox(BaseLightningModule):

    def __init__(self, params: TrainParameters) -> None:
        super(SceneSQPTSelBox, self).__init__(params, auto_optimize=False, auto_assign_devices=False)

    def _set_network(self, args):

        # parameters

        self.stage = from_meta(args, 'stage', default='d_K')
        self.dist_reg_loss_w = from_meta(args, 'reg_loss_w', default=0.5)
        self.pnp_loss_w = from_meta(args, 'pnp_loss_w', default=0.3)
        self.qp_sel_random = from_meta(args, 'qp_sel_random', default=False)
        self.is_reduce_dim = from_meta(args, 'is_reduce_dim', default=False)

        # configure the network ----------------------------------------------------------------------------------------
        self.q2r = SuperGlueMatcher(device=self.args.DEV_IDS[0])
        self.sqz = SceneSqueezerWithTestQueries(args=args, kypt_matcher=self.q2r,
                                                squeezer=SqueezerMixedTransformer(dim=[512, 256, 256, 256, 512], fix_encoder=True, fix_d_decoder=True))
        solver_type = from_meta(args, 'qp_solver', default='cvxpylayer')
        # self.pt_sel = PointSelection(args, debug=self.args.DEBUG,
        #                              solver_type=solver_type)
        self.pt_sel = PointSelection(args, in_ch=512 + 4, debug=self.args.DEBUG)

        if self.verbose_mode:
            notice_msg('Solver Type: %s' % (solver_type))

        # reduce dimension of features ---------------------------------------------------------------------------------
        if self.is_reduce_dim:
            self.quantizater = SoftQuant(encoder_dims=args['reduce_dim_list'], decoder_dims=args['increase_dim_list'])
            notice_msg('The dimension of feature is reduced to %d' % args['reduce_dim_list'][-1], obj=self)
        else:
            notice_msg('The dimension of feature is not reduced', obj=self) 
        
        self.dual_matcher = SuperGlueGNNMatcher(init_with_weights=self.q2r)
        
        # loss functions -----------------------------------------------------------------------------------------------
        self.pnp_loss = FastPnPLoss(args, pt_sel_thres=14)

    def _instance_scripts(self):
        return [self, self.q2r, self.sqz, self.pnp_loss, self.dual_matcher, self.quantizater, './train.sh']

    def _instance_devices(self):
        return {self.dev_ids[0]: [self.q2r, self.sqz, self.pt_sel, self.dual_matcher, self.quantizater]}

    def configure_optimizers(self):
        if self.args.AUX_CFG_DICT['stage'] == 'dual_matcher':
            params = self.dual_matcher.parameters()
        elif self.args.AUX_CFG_DICT['stage'] == 'only_d':
            params = self.q2r.parameters()
        elif self.args.AUX_CFG_DICT['stage'] == 'soft_quant':
            params = self.quantizater.parameters()
        else:
            params = self.parameters()
        return torch.optim.Adam(params, lr=self.args.START_LR)

    def load_from(self, ckpt_paths: dict):
        if 'instance' in ckpt_paths and ckpt_paths['instance'] != '' and Path(ckpt_paths['instance']).exists():
            self.load_state_dict(torch.load(ckpt_paths['instance'], map_location='cpu')['state_dict'],
                                 strict=False)
            notice_msg('model loaded from %s' % ckpt_paths['instance'], obj=self)
            
        if 'quantizater' in ckpt_paths and ckpt_paths['quantizater'] != '' and Path(ckpt_paths['quantizater']).exists():
            model_str = 'quantizater'
            state_dict = torch.load(ckpt_paths['quantizater'], map_location='cpu')['state_dict']
            quant_state_dict = {k[len(model_str) + 1:]: v for k, v in state_dict.items() if model_str in k}
            self.quantizater.load_state_dict(quant_state_dict, strict=True)
            notice_msg('[quantizater] overloaded from %s' % ckpt_paths['quantizater'], obj=self)


    """ Pipeline -------------------------------------------------------------------------------------------------------
    """

    def squeeze_scene_pts(self, vr_in: Tuple, anchor_in: Tuple, learnt_kernel=False):
        vr_metas, vr_pt2d = vr_in
        ar_metas, ar_pt2d, ar_pt3d = anchor_in

        (log_var, kernel_feats), (r_xyz, r_feats), _ = \
            self.sqz.forward(vr_metas, vr_pt2d, ar_metas, ar_pt2d, ar_pt3d)
        N = log_var.shape[1]

        # generate kernel: (fixed spatial kernel + learnt kernel)
        r_kernel = self.pt_sel.get_kernel(ar_pt3d.xyz, kernel_feats[0], spatial_only=not learnt_kernel)

        return log_var.view(N), r_kernel.view(N, N), r_xyz, r_feats


    def loss(self, query, anchor, log_var, alpha, pred_q2r_regs, q_indices, args=None):
        cur_dev = torch.cuda.current_device()
        q_metas, q_pt2d = query
        r_metas, r_pt2d, r_pt3d = anchor
        pred_r2q_dist = r2q_reproj_dist(q_metas, q_pt2d, r_pt3d, pred_q2r_regs)

        # distinctive loss ---------------------------------------------------------------------------------------------
        d_total_loss, dist_p_loss, dist_reg_loss = distinctive_loss(pred_r2q_dist, log_var.view(1, -1),
                                                                    reg_loss_w=self.dist_reg_loss_w)

        # qp selection loss
        pnp_reproj_losses, pnp_outliers_losses = [], []
        inlier_ratios = []
        max_q, count = from_meta(args, 'pnp_loss_max_quires', default=2), 0
        for q_i in q_indices:
            if count >= max_q:
                break

            pred_r2q = torch.from_numpy(extract_matches_r2q(pred_q2r_regs, q_i))

            outlier_loss, h_pnp_err, rl2q_2d_err, inlier_ratio = self.pnp_loss.forward(r_pt3d.xyz, alpha,
                                                                                       q_pt2d.uv[q_i],
                                                                                       q_metas.K[q_i],
                                                                                       q_metas.Tcws[q_i],
                                                                                       q_metas.dims[q_i], pred_r2q)

            if rl2q_2d_err is not None and torch.isnan(rl2q_2d_err).sum() == 0:
                print('ReprojLoss: %.2f, Inlier Ratio: %.2f' % (h_pnp_err.mean().item(), inlier_ratio.item()))
                pnp_outliers_losses.append(outlier_loss.view(-1).to(cur_dev))
                pnp_reproj_losses.append(h_pnp_err.view(-1).to(cur_dev))
                inlier_ratios.append(inlier_ratio.to(cur_dev).item())
                count += 1

        if len(pnp_reproj_losses) > 0:
            pnp_reproj_losses = torch.cat(pnp_reproj_losses, dim=0).mean().cpu()
            pnp_outliers_losses = torch.cat(pnp_outliers_losses, dim=0).mean().cpu()
            inlier_ratios = sum(inlier_ratios) / len(inlier_ratios)
        else:
            pnp_reproj_losses, pnp_outliers_losses, inlier_ratios = 0, 0, 0

        return dist_p_loss.mean(), dist_reg_loss.mean(), pnp_reproj_losses, pnp_outliers_losses, inlier_ratios

    """ Logs -----------------------------------------------------------------------------------------------------------
    """

    def log_visualization(self, batch_input, res_dict, vis_q2r=False):
        if self.logger is None:
            return

        q_imgs, _, q_info, rp_imgs, rp_info = batch_input[:5]

        # visualize distinctive score
        if 'log_var' in res_dict:
            d_scores = asnumpy(self.sqz.logvar2score(res_dict['log_var']))
            vis_dist_s = sq_vis.plot_sel_ref_pts(rp_imgs, rp_info, d_scores, args={'pt_radius': 5})
            LightningLogger.add_image(self.logger.experiment, vis_dist_s, name='dist_pt', step=self.global_step)

        # visualize selected points
        if 'alpha' in res_dict:
            vis_alpha = sq_vis.plot_sel_ref_pts(rp_imgs, rp_info, asnumpy(res_dict['alpha']), args={'pt_radius': 5})
            LightningLogger.add_image(self.logger.experiment, vis_alpha, name='alpha', step=self.global_step)

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

    # train with fixed encoder/decoder of dist_score -------------------------------------------------------------------
    def forward_dist_only(self, input):
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

        # print('points: %d, num_frames: q:%d, v:%d, a:%d' % (r_pt3d.xyz.shape[0], 
        #                                                 q_metas.num_frames(), vr_metas.num_frames(), r_metas.num_frames()))

        # step 1: build scene representation and squeeze the points ----------------------------------------------------
        res_dict = dict()
        with torch.cuda.device(self.device_of(self.sqz)) as _:
            log_var, r_kernel, r_xyz, r_feats = \
                self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d), anchor_in=(r_metas, r_pt2d, r_pt3d))
            if log_var is None:
                return None, None, None

            # dist_score = torch.exp(- log_var).view(-1)
            res_dict['log_var'] = log_var
            
        # step 3: registering query to references ----------------------------------------------------------------------
        with torch.cuda.device(self.device_of(self.q2r)) as _, torch.no_grad() as _:
            q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)
            q2r_scores = register_multi_q2r(self.q2r, q_sp_feats, r_feats)
            q2r_pred_matches = [self.q2r.get_matches(s, optimal_transport=True) for s in q2r_scores]

            for q_i, (score, pred_match) in enumerate(zip(q2r_scores, q2r_pred_matches)):
                res_dict[q_i] = {'P': score, 'matches': extract_matches(pred_match)}

        # step 4: optimize ---------------------------------------------------------------------------------------------
        with torch.cuda.device(self.device_of(self.pt_sel)):
            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
            pred_r2qs = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})

            if self.training:
                optimizer = self.optimizers()
                optimizer.zero_grad()

            # optimize -------------------------------------------------------------------------------------------------
            pred_r2q_dist = r2q_reproj_dist(q_metas, q_pt2d, r_pt3d, pred_r2qs)

            # distinctive loss
            d_total_loss, d_p_loss, d_reg_loss = distinctive_loss(pred_r2q_dist, log_var.unsqueeze(0), 
                                                                  reg_loss_w=self.dist_reg_loss_w)
            d_total_loss = d_total_loss.mean()

            if self.training:
                d_total_loss.mean().backward()
                optimizer.step()

        return {'total_loss': d_total_loss.item(),
                'p_loss': d_p_loss.mean(),
                'reg_loss': d_reg_loss.mean()}, res_dict, q_idx

    # train d and kernel -----------------------------------------------------------------------------------------------
    def forward(self, input, return_type='train'):
        fq_imgs, fq_segs, fq_info, rp_imgs, rp_info = input[:5]

        # parse the dict
        fq_metas, fq_pt2d, _ = dict2obs(fq_info)
        r_metas, r_pt2d, r_pt3d = dict2obs(rp_info)

        if r_pt3d.num_pts() < self.pt_sel.qp_num_pts or r_pt3d.num_pts() > 3500 or fq_metas.num_frames() < 2:
            return None, None, None

        # split the verification, query set
        q_idx, vr_idx = asnumpy(fq_info['q_idx']), asnumpy(fq_info['vr_idx'])
        q_metas, q_pt2d = sel_subset_clip(fq_metas, q_idx), sel_subset_obs2d(fq_pt2d, q_idx)
        vr_metas, vr_pt2d = sel_subset_clip(fq_metas, vr_idx), sel_subset_obs2d(fq_pt2d, vr_idx)

        # step 1: build scene representation and squeeze the points ----------------------------------------------------
        res_dict = dict()
        with torch.cuda.device(self.device_of(self.sqz)) as _:
            log_var, r_kernel, r_xyz, r_feats = \
                self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d), anchor_in=(r_metas, r_pt2d, r_pt3d), learnt_kernel=True)
            if log_var is None:
                return None, None, None
            log_var = log_var.detach()
            dist_score = torch.exp(- log_var).view(-1)
            res_dict['log_var'] = log_var

        # step 2: select the point by qp -------------------------------------------------------------------------------
        with torch.cuda.device(self.device_of(self.pt_sel)):
            cur_dev = torch.cuda.current_device()
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
            energy, quad_form, linear_form = self.pt_sel.get_qp_energy(r_kernel, dist_score, alpha, sel_idx=sel_pts_idx)
            print('Energy: %f, quad: %f, linear: %f' % (energy.item(), quad_form.item(), linear_form.item()))
            alpha /= self.pt_sel.get_pt_sel_thres()
            res_dict['alpha'] = alpha

        # step 3: registering query to references ----------------------------------------------------------------------
        with torch.cuda.device(self.device_of(self.q2r)) as _, torch.no_grad() as _:
            q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)
            q2r_scores = register_multi_q2r(self.q2r, q_sp_feats, r_feats)
            q2r_pred_matches = [self.q2r.get_matches(s, optimal_transport=True) for s in q2r_scores]

            for q_i, (score, pred_match) in enumerate(zip(q2r_scores, q2r_pred_matches)):
                res_dict[q_i] = {'P': score, 'matches': extract_matches(pred_match)}

        # step 4: optimize ---------------------------------------------------------------------------------------------
        with torch.cuda.device(self.device_of(self.pt_sel)):
            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
            pred_r2qs = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})

            if return_type == 'debug':
                q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
                pred_r2qs = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})
                return (r_metas, r_pt2d, r_pt3d, q_metas, q_pt2d, dist_score, pred_r2qs, r_kernel, res_dict, q_idx)

            if self.training:
                optimizer = self.optimizers()
                optimizer.zero_grad()

            # optimize -------------------------------------------------------------------------------------------------
            dist_p_loss, dist_reg_loss, pnp_hypo_losses, pnp_outliers_losses, inlier_ratios = \
                self.loss(query=(q_metas, q_pt2d), anchor=(r_metas, r_pt2d, r_pt3d),
                          log_var=log_var, alpha=alpha, pred_q2r_regs=pred_r2qs, q_indices=q_indices)

            total_loss = pnp_hypo_losses + pnp_outliers_losses
            # total_loss = dist_p_loss + self.dist_reg_loss_w * dist_reg_loss +  \
            #  self.pnp_loss_w * (pnp_hypo_losses + pnp_outliers_losses)

            if self.training and total_loss != 0:
                total_loss.backward()
                optimizer.step()

        return {'total_loss': total_loss,
                'p_loss': dist_p_loss,
                'reg_loss': dist_reg_loss,
                'reproj_loss': pnp_hypo_losses,
                'outliers_loss': pnp_outliers_losses,
                'inliers_percent': inlier_ratios,
                'tao': torch.exp(self.pt_sel.tao_o).item()}, res_dict, q_idx


    def training_step(self, batch, batch_idx):
        if self.stage == 'dual_matcher':
            loss, res_dict, _ = forward_dual_matcher(self, batch)
        elif self.stage == 'soft_quant':
            loss, res_dict, _ = forward_soft_quant(self, batch)
        elif self.stage == 'only_d':
            loss, res_dict, _ = self.forward_dist_only(batch)
        elif self.stage == 'd_K':
            loss, res_dict, _ = self.forward(batch)

        if loss is None:
            return torch.zeros(1, requires_grad=True).to(self.dev_ids[0])

        if 'reproj_loss' not in loss or loss['reproj_loss'] != 0:
            self.log_dict({'train/' + k: l for k, l in loss.items()}, prog_bar=True)

        # if self.on_visualize() and res_dict is not None:
        #     self.log_visualization(batch, res_dict)

        if self.global_step % 20 == 0 and self.logger is not None and res_dict is not None:  # histogram
            d_scores = asnumpy(self.sqz.logvar2score(res_dict['log_var']))
            LightningLogger.add_hist(self.logger.experiment, d_scores, name='d_scores', step=self.global_step)

        return loss['total_loss']

    def validation_step(self, batch, batch_idx):

        if self.stage == 'dual_matcher':
            loss, res_dict, _ = forward_dual_matcher(self, batch)
        elif self.stage == 'soft_quant':
            loss, res_dict, _ = forward_soft_quant(self, batch)            
        elif self.stage == 'only_d':
            loss, res_dict, _ = self.forward_dist_only(batch)
        elif self.stage == 'd_K':
            loss, res_dict, _ = self.forward(batch)

        if loss is None:
            return torch.zeros(1, requires_grad=True).to(self.dev_ids[0])
        
        if 'reproj_loss' not in loss or loss['reproj_loss'] != 0:
            self.log_dict({'valid/' + k: l for k, l in loss.items()}, prog_bar=False)

        # if self.on_visualize() and res_dict is not None:
        #     self.log_visualization(batch, res_dict)
        
        if self.global_step % 20 == 0 and self.logger is not None and res_dict is not None:  # histogram
            d_scores = asnumpy(self.sqz.logvar2score(res_dict['log_var']))
            LightningLogger.add_hist(self.logger.experiment, d_scores, name='d_scores', step=self.global_step)
