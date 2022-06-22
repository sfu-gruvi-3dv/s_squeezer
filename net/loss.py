import torch
import torch.nn as nn
import numpy as np
from core_3dv.camera_operator_gpu import transform_points, pi
from core_dl.torch_ext import index_of_elements, multi_index_select, ravel_multi_index



def exp_loss(all_matches: torch.Tensor, scores: torch.Tensor, eps=1e-9):
    assert all_matches.ndim == 2 and scores.ndim == 2
    assert all_matches.shape[1] == 2
    s = multi_index_select(scores, all_matches)
    return -torch.log(s + eps)


class DiffBinarizer(nn.Module):
    """ Differeintial version of binarization: https://arxiv.org/pdf/1911.08947.pdf
    """
    def __init__(self, init_threshold=0.5, k=50):
        super(DiffBinarizer, self).__init__()
        self.t = init_threshold
        self.k = k

    def forward(self, prob):
        return 1.0 / (1.0 + torch.exp(-self.k * (prob - self.t)))

    def __repr__(self):
        return self.__class__.__name__ + '(' + 't=' + '[{}]'.format(self.t) + ', ' + 'k=' + str(self.k) + ')'


class MSE_VAR(nn.Module):
    def __init__(self, var_weight):
        super(MSE_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, var = results['mean'], results['var']
        var = self.var_weight * var

        loss1 = torch.mul(torch.exp(-var), (mean - label) ** 2)
        loss2 = var
        loss = .5 * (loss1 + loss2)
        return loss.mean()


def distinctive_loss(pred_r2q_rdist: dict, log_var: torch.Tensor, dist_thres=6.0, reg_loss_w=1.0):
    cur_dev = torch.cuda.current_device()
    log_var = log_var.to(cur_dev)
    _, num_ref_pts = log_var.shape

    rpj_obs = torch.zeros(num_ref_pts)
    for r in pred_r2q_rdist.keys():
        rp = np.asarray(pred_r2q_rdist[r]).ravel()
        rp_valids = rp < dist_thres
        rpj_obs[r] = 1 - (rp_valids.sum() / rp_valids.shape[0])

    # compute loss
    loss1 = torch.exp(-log_var) * (rpj_obs.to(cur_dev))
    loss2 = reg_loss_w * log_var

    valid_r_idx = torch.Tensor([r for r in pred_r2q_rdist.keys()]).long()
    loss1 = loss1.view(-1)[valid_r_idx]
    loss2 = loss2.view(-1)[valid_r_idx]
    total_loss = (loss1 + loss2)
    return total_loss, loss1, loss2

def main():
    data = torch.load('dbg/sample_data.pt')

    q_imgs  = data['q_imgs']
    q_info  = data['q_info']
    rp_imgs = data['rp_imgs']
    rp_info = data['rp_info']
    matches = data['matches']
    q_idx = data['rand_q_id']

    reproj_error = reprojection_error(rp_info, q_info, matches, q_idx)
    print(reproj_error.tolist())


if __name__ == '__main__':
    main()
