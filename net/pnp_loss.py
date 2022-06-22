import torch
import torch.nn as nn

import cv2
import random

from core_io.meta_io import from_meta
import core_3dv.camera_operator_gpu as cam_opt_gpu
from dataset.common.base_data_source import ClipMeta, Pt2dObs


class RandomizedPnPLoss(nn.Module):
    """ randomly selects sets of 4-points to calculate PnP loss
    """
    def __init__(self, args: dict, pt_sel_thres: float):
        super(RandomizedPnPLoss, self).__init__()
        self.args = args
        self.pt_sel_thres = pt_sel_thres
        self.num_samples = from_meta(args, 'pnp_loss_samples', default=5)

    def forward(
        self,
        r_xyz: torch.Tensor, pt_sel_dist: torch.Tensor,
        q_meta: ClipMeta, q_pt2d_obs: Pt2dObs, q_idx: int,
        r_meta: ClipMeta, r_pt2d_obs: Pt2dObs, r_idx: int,
        sel_matches: torch.Tensor,
        **kwargs
    ) -> dict:
        """
        @param xyz: (N, 3) Point coords
        @param pt_sel_dist: (N,) Points distribution (alpha)
        """
        cur_dev = torch.cuda.current_device()

        q_pos_2d = q_pt2d_obs.uv[q_idx].view(-1, 2)
        r_pos_2d = r_pt2d_obs.uv[r_idx].view(-1, 2)
        r_obs3d = r_pt2d_obs.obs3d_ids[r_idx].view(-1)

        qp_mask_3d = (pt_sel_dist > self.pt_sel_thres)
        qp_mask_2d = qp_mask_3d[r_obs3d][sel_matches[:, 1]]

        # r_xyz: 3D frame, r_obs3d: ref 2d frame, sel_matches: q2r matches frame
        r_3d_sel_qp = r_xyz[r_obs3d][sel_matches[:, 1]][qp_mask_2d]
        q_2d_sel_qp = q_pos_2d[sel_matches[:, 0]][qp_mask_2d]
        r_2d_sel_qp = r_pos_2d[sel_matches[:, 1]][qp_mask_2d]

        if r_3d_sel_qp.shape[0] < 4:
            print('not enough points selected: {}'.format(r_3d_sel_qp.shape[0]))
            zero_tensor = torch.tensor(
                0.0, requires_grad=True, device=cur_dev
            )
            return {
                'outliers_loss': zero_tensor,
                'reproj_loss': zero_tensor,
                'pnp_failure_loss': zero_tensor,
            }

        rpj_3d_local = cam_opt_gpu.transpose(
                q_meta.Tcws[q_idx][:3, :3],
                q_meta.Tcws[q_idx][:3, -1],
                r_3d_sel_qp.unsqueeze(0)
        )
        rpj_2d_pos, _ = cam_opt_gpu.pi(q_meta.K[q_idx], rpj_3d_local)
        rpj_err = torch.norm(rpj_2d_pos.squeeze(0) - q_2d_sel_qp, dim=-1)
        outliers_loss = torch.mean(rpj_err) * torch.prod(
            pt_sel_dist[r_obs3d][sel_matches[:, 1]][qp_mask_2d]
        )

        # TODO: debug only
        # from visualizer.corres_plot import plot_images, plot_matches
        # from core_dl.torch_vision_ext import tensor_to_vis
        # import matplotlib.pyplot as plt
        # from time import time

        # timestamp = str(int(time()))
        # print('timestamp', timestamp)

        # q_img = tensor_to_vis(kwargs['q_imgs'])[q_idx]
        # r_img = tensor_to_vis(kwargs['r_imgs'])[r_idx]

        # plot_images([q_img, r_img])
        # plot_matches(
        #     q_pos_2d[sel_matches[:, 0]].cpu() * 0.5,
        #     r_pos_2d[sel_matches[:, 1]].cpu() * 0.5,
        #     a=0.2
        # )
        # plt.savefig('dbg/{}_all_matches.png'.format(timestamp))
        # plt.close()

        # plot_images([q_img, r_img])
        # plot_matches(q_2d_sel_qp.cpu() * 0.5, r_2d_sel_qp.cpu() * 0.5, a=0.2)
        # plt.savefig('dbg/{}_selected_matches.png'.format(timestamp))
        # plt.close()

        # plot_images([q_img, r_img])
        # inliers = (rpj_err <= 12)
        # plot_matches(q_2d_sel_qp[inliers].cpu() * 0.5, r_2d_sel_qp[inliers].cpu() * 0.5, a=0.2)
        # plt.savefig('dbg/{}_inliers.png'.format(timestamp))
        # plt.close()

        # plot_images([q_img, r_img])
        # outliers = (rpj_err > 12)
        # plot_matches(q_2d_sel_qp[outliers].cpu() * 0.5, r_2d_sel_qp[outliers].cpu() * 0.5, a=0.2)
        # plt.savefig('dbg/{}_outliers.png'.format(timestamp))
        # plt.close()

        reproj_loss = 0.0
        pnp_failure_loss = 0.0
        for sample_idx in range(self.num_samples):
            pnp_sample_indices = random.sample(range(r_3d_sel_qp.shape[0]), 3)
            r_3d_sample = r_3d_sel_qp[pnp_sample_indices]
            q_2d_sample = q_2d_sel_qp[pnp_sample_indices]
            r_2d_sample = r_2d_sel_qp[pnp_sample_indices]

            # plot_images([q_img, r_img])
            # plot_matches(q_2d_sample.cpu() * 0.5, r_2d_sample.cpu() * 0.5, a=0.6)
            # plt.savefig('dbg/%s_%d.png' % (timestamp, sample_idx))
            # plt.close()

            pnp_success, R_exp, t = cv2.solvePnP(
                r_3d_sample.detach().cpu().numpy(),
                q_2d_sample.detach().cpu().numpy(),
                q_meta.K[q_idx].detach().cpu().numpy(),
                None,
                flags=cv2.SOLVEPNP_SQPNP
            )
            if not pnp_success:
                # penalize failure
                pnp_failure_loss += self.pnp_failure_weight * torch.prod(
                    pt_sel_dist[r_obs3d][sel_matches[:, 1]][qp_mask_2d][pnp_sample_indices]
                )
                continue

            R, _ = cv2.Rodrigues(R_exp)
            R = torch.from_numpy(R).to(r_3d_sample.device).type(r_3d_sample.dtype)
            t = torch.from_numpy(t).to(r_3d_sample.device).type(r_3d_sample.dtype)

            rpj_3d_local = cam_opt_gpu.transpose(
                R.unsqueeze(0), t.view(1, 3), r_3d_sel_qp.unsqueeze(0)
            )
            rpj_2d_pos, _ = cam_opt_gpu.pi(q_meta.K[q_idx], rpj_3d_local)
            rpj_err = torch.norm(rpj_2d_pos.squeeze(0) - q_2d_sel_qp, dim=-1)
            reproj_loss += torch.mean(rpj_err) * torch.prod(
                pt_sel_dist[r_obs3d][sel_matches[:, 1]][qp_mask_2d][pnp_sample_indices]
            )
            # print(sample_idx, reproj_loss, torch.mean(rpj_err), pnp_failure_loss)

        return {
            'outliers_loss': outliers_loss,
            'reproj_loss': reproj_loss,
            'pnp_failure_loss': pnp_failure_loss,
        }
