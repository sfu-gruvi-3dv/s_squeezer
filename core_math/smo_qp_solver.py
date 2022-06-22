import torch
import numpy as np
from time import time

from core_io.serialize import load_pickle

import matplotlib.pyplot as plt


class SMO_QP:
    def __init__(
            self, compression_ratio=0.1,
            num_batched_pairs=100, num_iters=50000,
            debug=False
    ):
        self.compression_ratio = compression_ratio
        self.num_iters = num_iters
        self.num_batched_pairs = num_batched_pairs
        self.debug = debug

    def __call__(self, rbf_kernel: torch.Tensor, dist_score: torch.Tensor) -> torch.Tensor:
        return (solve_qp(
            dist_score, rbf_kernel,
            compression_ratio=self.compression_ratio,
            num_iters=self.num_iters,
            num_batched_pairs=self.num_batched_pairs,
            debug=self.debug
        ),)

    def forward(self, rbf_kernel: torch.Tensor, dist_score: torch.Tensor) -> torch.Tensor:
        return self(rbf_kernel, dist_score)


def pair_selection(m, alpha, nu, num_pairs=1):
    threshold = 1.0 / (m * nu)
    # i, j = np.random.choice(range(m), 2, replace=False).tolist()

    zeros = torch.nonzero(alpha < threshold).view(-1)
    nonzeros = torch.nonzero(alpha).view(-1)
    all_pts = torch.unique(torch.cat((zeros, nonzeros)))
    max_pairs = min(2*num_pairs, all_pts.shape[0], zeros.shape[0])

    i = np.random.choice(range(all_pts.shape[0]), max_pairs, replace=False)
    i = all_pts[i]

    j = np.random.choice(range(zeros.shape[0]), max_pairs, replace=False)
    j = zeros[j]

    valid_pairs = torch.nonzero(i != j).view(-1)
    i = i[valid_pairs][:num_pairs]
    j = j[valid_pairs][:num_pairs]
    if i.numel() < num_pairs:
        print('[SMO] not enought valid pairs')

    return i, j


@torch.no_grad()
def solve_qp(
    dist_scores, kernel, compression_ratio=0.1,
    num_iters=10000, num_batched_pairs=100, debug=False
):
    alpha = get_initial_solution(dist_scores, compression_ratio)
    m = dist_scores.shape[0]

    K_energys = []
    d_energys = []
    total_energys = []

    for batch_idx in range(0, num_iters, num_batched_pairs):
        pairs_i, pairs_j = pair_selection(m, alpha, compression_ratio, num_batched_pairs)
        pairs = torch.stack((pairs_i, pairs_j), dim=-1)

        for pair in pairs.unbind(0):
            i = pair[0].item()
            j = pair[1].item()
            if i == j:
                print('[SMO] incorrect pair')
                continue

            delta = alpha[i] + alpha[j]
            # assert(delta <= (2 / (m * compression_ratio)))
            alpha[i] = compute_probability(
                dist_scores, kernel, i, j, alpha, compression_ratio
            )
            alpha[j] = delta - alpha[i]

            if debug:
                K_energy = alpha.view(1, m).mm(kernel).mm(alpha.view(m, 1)).item()
                d_energy = -torch.dot(dist_scores, alpha).item()
                total_energy = K_energy + d_energy

                K_energys.append(K_energy)
                d_energys.append(d_energy)
                total_energys.append(total_energy)

    if debug:
        plt.subplot(3, 1, 1)
        plt.plot(K_energys, label='K_energy')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(d_energys, label='d_energy')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(total_energys, label='total_energy')
        plt.legend()
        plt.savefig('dbg/smo_qp.png')

        plt.close()

    return alpha


def compute_probability(dist_scores, kernel, i, j, init_soln, compression_ratio):
    # short-hand symbols
    nu = compression_ratio
    tau = 1.0 # distinctiveness_weight
    m = dist_scores.shape[0]
    d = dist_scores
    K = kernel
    alpha = init_soln
    delta = alpha[i] + alpha[j]
    max_prob = 1.0/int(nu * m)

    # equation 12
    dot_prod1 = torch.dot(alpha, K[i, :]) - (alpha[i] * K[i, i]) - (alpha[j] * K[i, j])
    dot_prod2 = torch.dot(alpha, K[j, :]) - (alpha[i] * K[j, i]) - (alpha[j] * K[j, j])
    T = tau * (d[i] - d[j]) - (2 * dot_prod1) + (2 * dot_prod2)

    # equation 11
    alpha_i= 1.0 / 2.0 * T / (2 * (1 - K[i, j]) + delta)

    # equation 13
    alpha_i = max(0, min(min(max_prob, delta.item()), alpha_i.item()))
    alpha_j = (delta - alpha_i).item()

    assert(alpha_i <= (max_prob + 1e-6))
    assert(alpha_i <= delta)

    if alpha_j > (max_prob + 1e-6) or alpha_j > delta:
        # print(
        #     'discarding alphas:', alpha_i, alpha_j,
        #     'reverting to:', alpha[i].item(), alpha[j].item()
        # )
        alpha_i = alpha[i]
        alpha_j = alpha[j]

    assert(alpha_j <= (max_prob + 1e-6))
    assert(alpha_j <= delta)

    return alpha_i


def get_initial_solution(dist_scores, compression_ratio):
    k = int(dist_scores.shape[0] * compression_ratio)
    soln = torch.zeros_like(dist_scores)

    # top_k = np.random.choice(range(dist_scores.shape[0]), k, replace=False)
    _, top_k = torch.topk(dist_scores, k)

    soln[top_k] = 1.0 / k
    return soln


if __name__ == '__main__':
    qp_input = load_pickle('/tmp/dbg_qp.bin')
    qp_input = {
        'alpha': qp_input[0],
        'dist_scores': qp_input[1],
        'r_kernel': qp_input[2],
    }
    # qp_input = load_pickle('/tmp/qp_input.bin')
    dist_scores = qp_input['dist_scores'].detach().cpu()
    kernel = qp_input['r_kernel'].detach().cpu()
    cached_alpha = qp_input['alpha'].detach().cpu()
    m = dist_scores.shape[0]

    distinctiveness_weight = 0.1
    compression_ratio = 0.1
    max_qp_layer_feats = 300
    max_cvxpy_feats = 1000
    num_batched_pairs = 100

    def get_energy(alpha):
        m = alpha.numel()
        K_energy = alpha.view(1, m).mm(kernel[:m, :m]).mm(alpha.view(m, 1)).item()
        d_energy = -torch.dot(dist_scores[:m], alpha.view(-1)).item()
        total_energy = K_energy + (distinctiveness_weight * d_energy)
        print('K: {}, d: {}, total: {}'.format(K_energy, d_energy, total_energy))

    ########## Cached alpha ##########
    print('cached')
    get_energy(cached_alpha)

    ########## cvxpy_layer alpha ##########
    from net.qp_layer_cholesky import get_qp_layer
    from core_math.matrix_sqrt import sqrt_newton_schulz_autograd

    tic = time()
    qp_layer = get_qp_layer(max_qp_layer_feats, compression_ratio)

    with torch.no_grad():
        kernel_sqrt, _ = sqrt_newton_schulz_autograd(
            kernel.unsqueeze(0), numIters=15, dtype=kernel.dtype
        )
        qp_layer_alpha = qp_layer(
            kernel_sqrt[0, :max_qp_layer_feats, :max_qp_layer_feats],
            distinctiveness_weight * dist_scores[:max_qp_layer_feats, None]
        )[0]

    print('qp_layer time:', time() - tic)
    print('qp_layer')
    get_energy(qp_layer_alpha)

    ########## cvxpy alpha ##########
    from core_math.cvxpy_qp_solver import CVXPY_QP
    tic = time()
    cxvpy_qp = CVXPY_QP(max_cvxpy_feats, compression_ratio)
    cxvpy_alpha = cxvpy_qp(kernel, distinctiveness_weight * dist_scores)[0]
    print('cvxpy time:', time() - tic)
    print('cvxpy')
    get_energy(cxvpy_alpha)

    ########## SMO alpha ##########
    smo_qp = SMO_QP(compression_ratio, num_batched_pairs, debug=True)
    tic = time()
    smo_alpha = smo_qp(kernel, distinctiveness_weight * dist_scores)[0]
    print('smo time:', time() - tic)
    print('SMO')
    get_energy(smo_alpha)
    print(smo_alpha.sum().item(), smo_alpha.max().item(), smo_alpha.min().item())
    print(m, torch.nonzero(smo_alpha > (0.99 / (m * compression_ratio))).numel())

    def get_top_points(alpha):
        # return alpha.detach().cpu().numpy()
        if len(alpha.shape) == 2:
            alpha = alpha.squeeze(1)
        selection = np.zeros(alpha.shape[0])
        k = int(compression_ratio * alpha.shape[0])
        _, top_k = torch.topk(alpha, k)
        selection[top_k.detach().cpu().numpy()] = 1.0
        return selection

    ######## Plot ########
    plt.subplot(2, 2, 1)
    plt.plot(get_top_points(cached_alpha), label='cached')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(get_top_points(smo_alpha), label='smo')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(get_top_points(qp_layer_alpha), label='qp_layer')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(get_top_points(cxvpy_alpha), label='cxvpy')
    plt.legend()

    plt.savefig('dbg/smo_sq_soln.png')
    plt.close()
