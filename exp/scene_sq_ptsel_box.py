from itertools import chain
from torch.autograd.grad_mode import no_grad

import torch.cuda

import exp.scene_sq_visualizer as sq_vis
from core_dl.lightning_logger import LightningLogger
from core_dl.lightning_model import BaseLightningModule
from core_dl.train_params import TrainParameters
from dataset.common.split_scene import sel_subset_clip, sel_subset_obs2d
from net.dlt_pnp_loss import DLTPnPLoss
from net.fast_pnp_loss import FastPnPLoss
from net.qp_ptsel import PointSelection
from net.scene_fuser_sq import *
from core_dl.expr_ctx import ExprCtx

import time
from scipy.linalg import sqrtm

class SceneSQPTSelBox(BaseLightningModule):

    def __init__(self, params: TrainParameters) -> None:
        super(SceneSQPTSelBox, self).__init__(params, auto_optimize=False, auto_assign_devices=False)

    def _set_network(self, args):

        # parameters
        self.reg_loss_w = from_meta(args, 'reg_loss_w', default=0.5)
        self.train_sqz = from_meta(args, 'train_sqz', default=False)
        self.qp_sel_random = from_meta(args, 'qp_sel_random', default=False)

        # configure the network
        self.q2r = SuperGlueMatcher(device=self.args.DEV_IDS[0])
        self.sqz = SceneSqueezerWithTestQueries(args=args, kypt_matcher=self.q2r)
        self.pt_sel = PointSelection(args, in_ch=512 + 4, debug=self.args.DEBUG)

        # self.pnp_loss = DLTPnPLoss(args)
        self.pnp_loss = FastPnPLoss(args, pt_sel_thres=14)

        # do not train squeezer and superglue
        if not self.train_sqz:
            for param in chain(self.q2r.parameters(), self.sqz.parameters()):
                param.requires_grad = False

    def _instance_scripts(self):
        return [self, self.q2r, self.sqz, self.pnp_loss, './train.sh']

    def _instance_devices(self):
        return {self.dev_ids[0]: [self.q2r, self.sqz, self.pt_sel]}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.START_LR)

    def load_from(self, ckpt_paths: dict):
        if 'instance' in ckpt_paths and ckpt_paths['instance'] != '' and Path(ckpt_paths['instance']).exists():
            self.load_state_dict(torch.load(ckpt_paths['instance'], map_location='cpu')['state_dict'],
                                 strict=False)
            notice_msg('model loaded from %s' % ckpt_paths['instance'], obj=self)

    """ Pipeline -------------------------------------------------------------------------------------------------------
    """

    def squeeze_scene_pts(self, vr_in: Tuple, anchor_in: Tuple):
        vr_metas, vr_pt2d = vr_in
        ar_metas, ar_pt2d, ar_pt3d = anchor_in

        log_var, (r_xyz, r_feats), interms = self.sqz.forward(vr_metas, vr_pt2d, ar_metas, ar_pt2d, ar_pt3d)

        return log_var, r_xyz, r_feats, interms

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
    def squeeze(self, input):
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

        # step 1: build scene representation and squeeze the points
        res_dict = dict()
        with torch.cuda.device(self.device_of(self.sqz)) as _, torch.no_grad() as _:
            log_var, r_xyz, r_feats, r_interm = \
                self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d), anchor_in=(r_metas, r_pt2d, r_pt3d))
            if log_var is None:
                return None, None, None

            res_dict['log_var'] = log_var

        # step 2: query to scene registering
        with torch.cuda.device(self.device_of(self.q2r)) as _, torch.no_grad() as _:
            q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)
            q2r_scores = register_multi_q2r(self.q2r, q_sp_feats, r_feats)
            q2r_pred_matches = [self.q2r.get_matches(s, optimal_transport=True) for s in q2r_scores]

            for q_i, (score, pred_match) in enumerate(zip(q2r_scores, q2r_pred_matches)):
                res_dict[q_i] = {'P': score, 'matches': extract_matches(pred_match)}

        # step 3: get kernel
        with torch.cuda.device(self.device_of(self.pt_sel)) as _:
            cur_dev = torch.cuda.current_device()
            B, M, _ = r_interm['in_feats'].shape
            dist_score = torch.exp(- log_var).to(cur_dev).view(B, M, 1)
            input_feats = torch.cat([r_interm['in_feats'], dist_score], dim=-1).contiguous()
            r_kernel = self.pt_sel.get_distance_kernel(r_xyz)

        # step 4: run multiple sampling and optimize
        with torch.cuda.device(self.device_of(self.pt_sel)):
            cur_dev = torch.cuda.current_device()
            r_kernel = r_kernel.to(cur_dev)

            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
            pred_r2qs = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})

            dist_score = dist_score.to(cur_dev).view(-1)
            # r_kernel = Variable(r_kernel, requires_grad=True).to(cur_dev)

            if self.qp_sel_random:
                sel_pts_idx = np.arange(0, dist_score.shape[0])
                np.random.shuffle(sel_pts_idx)
            else:
                sel_pts_idx = asnumpy(torch.argsort(dist_score.view(-1), descending=True))

            sel_pts_idx = sel_pts_idx[:self.pt_sel.qp_num_pts]
            n_dist_score = dist_score.clone()
            n_dist_score.requires_grad = True
            alpha = self.pt_sel.sel_by_qp(r_kernel, n_dist_score, sel_idx=sel_pts_idx)
            energy, quad_form, linear_form = self.pt_sel.get_qp_energy(r_kernel, n_dist_score, alpha, sel_idx=sel_pts_idx)
            print('Energy: %f, quad: %f, linear: %f' % (energy.item(), quad_form.item(), linear_form.item()))
            alpha /= self.pt_sel.get_pt_sel_thres()
            res_dict['alpha'] = alpha.cpu().clone().detach()

        return dist_score, alpha, r_xyz, r_feats


    def forward(self, input, no_sel=False):
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

        # step 1: build scene representation and squeeze the points
        res_dict = dict()
        with torch.cuda.device(self.device_of(self.sqz)) as _, torch.no_grad() as _:
            log_var, r_xyz, r_feats, r_interm = \
                self.squeeze_scene_pts(vr_in=(vr_metas, vr_pt2d), anchor_in=(r_metas, r_pt2d, r_pt3d))
            if log_var is None:
                return None, None, None

            res_dict['log_var'] = log_var

        # step 2: query to scene registering
        with torch.cuda.device(self.device_of(self.q2r)) as _, torch.no_grad() as _:
            q_sp_feats = encode_sp_feats(self.q2r, q_metas, q_pt2d)
            q2r_scores = register_multi_q2r(self.q2r, q_sp_feats, r_feats)
            q2r_pred_matches = [self.q2r.get_matches(s, optimal_transport=True) for s in q2r_scores]

            for q_i, (score, pred_match) in enumerate(zip(q2r_scores, q2r_pred_matches)):
                res_dict[q_i] = {'P': score, 'matches': extract_matches(pred_match)}

        # step 3: get kernel
        with torch.cuda.device(self.device_of(self.pt_sel)) as _:
            cur_dev = torch.cuda.current_device()
            B, M, _ = r_interm['in_feats'].shape
            dist_score = torch.exp(- log_var).to(cur_dev).view(B, M, 1)
            input_feats = torch.cat([r_interm['in_feats'], dist_score], dim=-1).contiguous()
            r_kernel, r_k_feats = self.pt_sel.get_kernel(r_xyz, input_feats)
            if torch.sum(torch.isnan(r_kernel)).item() > 0:
                err_msg('Kernel has NAN')
                return None, None, None

        if no_sel:
            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
            pred_r2qs = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})
            dist_score = dist_score.to(cur_dev).view(-1)
            # ExprCtx().attr_dict['sample'] = (r_metas, r_pt2d, r_pt3d, q_metas, q_pt2d, dist_score, pred_r2qs)
            return (r_metas, r_pt2d, r_pt3d, q_metas, q_pt2d, dist_score, pred_r2qs, r_kernel, r_k_feats, res_dict, q_idx)

        # step 4: run multiple sampling and optimize
        with torch.cuda.device(self.device_of(self.pt_sel)):
            cur_dev = torch.cuda.current_device()
            r_kernel = r_kernel.to(cur_dev)

            q_indices = [k for k in res_dict.keys() if isinstance(k, int)]
            pred_r2qs = r2q({q: res_dict[q]['matches'].cpu() for q in q_indices})

            dist_score = dist_score.to(cur_dev).view(-1)
            # r_kernel = Variable(r_kernel, requires_grad=True).to(cur_dev)

            # random sample points for qp selection --------------------------------------------------------------------
            if self.qp_sel_random:
                sel_pts_idx = np.arange(0, dist_score.shape[0])
                np.random.shuffle(sel_pts_idx)
            else:
                sel_pts_idx = asnumpy(torch.argsort(dist_score.view(-1), descending=True))

            sel_pts_idx = sel_pts_idx[:self.pt_sel.qp_num_pts]
            n_dist_score = dist_score.clone()
            n_dist_score.requires_grad = True

            def solver_qp_np(K, d, compression_ratio=0.7, lambda_=1.0):
                import cvxpy as cp
                # K to make problem DPP https://www.cvxpy.org/tutorial/advanced/index.html
                # K_data = np.random.randn(num_features, num_features)
                num_features = K.shape[0]
                K_placeholder = cp.Parameter((num_features, num_features), PSD=True)
                # K_sqrt_placeholder = cp.Parameter((num_features, num_features))
                d_placeholder = cp.Parameter((num_features, 1))

                alpha = cp.Variable((num_features, 1), pos=True)
                constraints = [
                    cp.sum(alpha) == 1,
                    alpha >= 0,
                    alpha <= 1.0 / (compression_ratio * num_features)
                ]
                objective = cp.Minimize(
                    # this expression non DPP
                    cp.quad_form(alpha, K_placeholder) - lambda_ * (d_placeholder.T @ alpha)
                    # cp.sum_squares(K_sqrt_placeholder @ alpha) - lambda_ * (d_placeholder.T @ alpha)
                )

                tic = time.time()
                problem = cp.Problem(objective, constraints)
                print('creation:', time.time() - tic)

                return {
                    'problem': problem,
                    # 'K': K_placeholder,
                    'K_sqrt': K_sqrt_placeholder,
                    'd': d_placeholder,
                    'alpha': alpha
                }

            def solve_qp_np(solver, K, d):
                if isinstance(K, torch.Tensor):
                    K = asnumpy(K)
                if isinstance(d, torch.Tensor):
                    d = asnumpy(d.view(-1, 1))

                K_sqrt = sqrtm(K)
                solver['K_sqrt'].value = K_sqrt

                # solver['K'].value = K

                solver['d'].value = d

                tic = time.time()
                solver['problem'].solve(solver='OSQP')
                print('solve:', time.time() - tic)
                return solver['alpha'].value

            print('input', r_kernel.shape, n_dist_score.shape)
            solver = solver_qp_np(r_kernel, n_dist_score)
            alpha_all = solve_qp_np(solver, r_kernel, n_dist_score)
            alpha_all = solve_qp_np(solver, r_kernel, n_dist_score)
            alpha_all = solve_qp_np(solver, r_kernel, n_dist_score)
            alpha_all = solve_qp_np(solver, r_kernel, n_dist_score)
            print('output', alpha_all.shape)

            # select by qp ---------------------------------------------------------------------------------------------
            alpha = self.pt_sel.sel_by_qp(r_kernel, n_dist_score, sel_idx=sel_pts_idx)
            if alpha is None:
                err_msg('QP selection error')
                return None, None, None
            energy, quad_form, linear_form = self.pt_sel.get_qp_energy(r_kernel, n_dist_score, alpha, sel_idx=sel_pts_idx)
            print('Energy: %f, quad: %f, linear: %f' % (energy.item(), quad_form.item(), linear_form.item()))
            alpha /= self.pt_sel.get_pt_sel_thres()
            res_dict['alpha'] = alpha.cpu().clone().detach()

            if self.args.DEBUG:
                ExprCtx().attr_dict['sample'] = (r_metas, r_pt2d, r_pt3d, q_metas, q_pt2d, alpha, dist_score, pred_r2qs)

            # optimize -------------------------------------------------------------------------------------------------
            if self.training:
                optimizer = self.optimizers()
                optimizer.zero_grad()

            reproj_losses = []
            outliers_losses = []
            inlier_ratios = []
            max_q, count = 2, 0
            for q_i in q_indices:
                if count >= max_q:
                    break

                pred_r2q = torch.from_numpy(extract_matches_r2q(pred_r2qs, q_i))

                outlier_loss, h_pnp_err, rl2q_2d_err, inlier_ratio = self.pnp_loss.forward(r_pt3d.xyz, alpha, q_pt2d.uv[q_i], q_metas.K[q_i], q_metas.Tcws[q_i], q_metas.dims[q_i], pred_r2q)

                if rl2q_2d_err is not None and torch.isnan(rl2q_2d_err).sum() == 0:
                    print('ReprojLoss: %.2f, Inlier Ratio: %.2f' % (h_pnp_err.mean().item(), inlier_ratio.item()))
                    outliers_losses.append(outlier_loss.view(-1).to(cur_dev))
                    reproj_losses.append(h_pnp_err.view(-1).to(cur_dev))
                    inlier_ratios.append(inlier_ratio.to(cur_dev).item())
                    count += 1

            total_loss = 0
            if len(reproj_losses) > 0:
                reproj_losses = torch.cat(reproj_losses, dim=0).mean().cpu()
                outliers_losses = torch.cat(outliers_losses, dim=0).mean().cpu()
                inlier_ratios = sum(inlier_ratios) / len(inlier_ratios)
                total_loss = outliers_losses + reproj_losses
            else:
                total_loss = torch.zeros(1, requires_grad=True).cpu()
                reproj_losses, inlier_ratios = 0, 0
                outliers_losses = 0

            if self.training:
                total_loss.backward()
                #  grad = n_dist_score.grad
                optimizer.step()

        return {
                'total_loss': total_loss.item(),
                'reproj_loss': reproj_losses,
                'outliers_loss': outliers_losses,
                'inliers_percent': inlier_ratios,
                'lambda_w': self.pt_sel.lambda_weight.item()
               }, res_dict, q_idx

    def training_step(self, batch, batch_idx):
        loss, res_dict, _ = self.forward(batch)
        if loss is None:
            return torch.zeros(1, requires_grad=True).to(self.dev_ids[0])

        # log
        if loss['reproj_loss'] != 0:
            self.log_dict(loss, prog_bar=True)

        if self.on_visualize():
            self.log_visualization(batch, res_dict)

        if self.global_step % 20 == 0 and self.logger is not None:  # histogram
            d_scores = asnumpy(self.sqz.logvar2score(res_dict['log_var']))
            LightningLogger.add_hist(self.logger.experiment, d_scores, name='d_scores', step=self.global_step)

        return loss['total_loss']
