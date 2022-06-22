import torch
import torch.nn as nn

from core_dl.train_params import TrainParameters
from core_io.meta_io import from_meta

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from core_math.matrix_sqrt import sqrt_newton_schulz_autograd
import diffcp

def get_qp_layer(
        num_features, compression_ratio=0.7, distinctiveness_weight=0.5
):
    alpha = cp.Variable(num_features)
    # K_sqrt to make problem DPP https://www.cvxpy.org/tutorial/advanced/index.html
    K_sqrt = cp.Parameter((num_features, num_features))
    d = cp.Parameter((num_features, 1))
    constraints = [
        cp.sum(alpha) == 1,
        alpha >= 0,
        alpha <= 1.0 / compression_ratio / num_features
    ]

    # this expression non DPP
    # quad_term = alpha.T @ K @ alpha
    quad_term = cp.sum(K_sqrt @ alpha)

    linear_term = d.T @ alpha

    objective = cp.Minimize(quad_term - (1.0 * linear_term))
    problem = cp.Problem(objective, constraints)
    # problem.solve(solver=cp.ECOS)  # solve using ECOS

    cvxpylayer = CvxpyLayer(problem, parameters=[K_sqrt, d], variables=[alpha])
    return cvxpylayer


class QPLayer(nn.Module):
    def __init__(self, args: dict, debug=False):
        super(QPLayer, self).__init__()
        self.args = args
        self.debug = debug
        self._set_qp_layer()

    def _set_qp_layer(self) -> None:

        kpt_num = 500 if self.debug else 600

        self.qp_max_input_keypoints = from_meta(
            self.args, 'qp_max_input_keypoints', default=kpt_num
        )
        self.qp_compression_ratio = from_meta(
            self.args, 'qp_compression_ratio', default=0.7
        )
        self.qp_distinctiveness_weight = from_meta(
            self.args, 'qp_distinctiveness_weight', default=500.0
        )
        self.qp_sqrtm_iters = from_meta(
            self.args, 'qp_sqrtm_iters', default=10
        )
        self.qp_solver_max_iters = from_meta(
            self.args, 'qp_solver_max_iters', default=1000
        )

        # threshold of QP solution to be considered selected
        self.pt_sel_thres = 1. / (self.qp_max_input_keypoints * self.qp_compression_ratio)
        # small number subtracted to avoid rounding error invalidating all points
        self.pt_sel_thres -= 1e-6

        self.qp_layer = get_qp_layer(
            self.qp_max_input_keypoints, self.qp_compression_ratio,
            self.qp_distinctiveness_weight
        )

    @staticmethod
    def pad_rbf_kernel(rbf_kernel: torch.Tensor, pad_size: int) -> torch.Tensor:
        if pad_size == 0:
            return rbf_kernel

        pad_start = rbf_kernel.shape[1] - pad_size

        # rows/cols corresponding to padded elements should be zero
        rbf_kernel[:, pad_start:, :] = 0.0
        rbf_kernel[:, :, pad_start:] = 0.0

        # diagonal elements should always be 1
        # for rbf to be positive semi-definite
        diag_indices = torch.arange(pad_start, rbf_kernel.shape[1])
        rbf_kernel[:, diag_indices, diag_indices] = 1.0
        return rbf_kernel

    def forward(self, dist_score: torch.Tensor, rbf_kernel: torch.Tensor) -> torch.Tensor:
        assert(dist_score.shape[0] == rbf_kernel.shape[0])
        assert(dist_score.shape[0] == rbf_kernel.shape[1])

        # TODO:
        # if the points are padded or clipped, the compression ratio won't hold
        # can select more points (if clipped) or less points (if padded)
        # the QP layer solution is not ranked, selecting top-k scores won't help
        # fix it!
        num_keypoints = min(dist_score.shape[0], self.qp_max_input_keypoints)

        # TODO: random selection instead of clip
        # clip to max num of keypoints
        dist_score_subset = dist_score[:num_keypoints]
        rbf_kernel_subset = rbf_kernel[:num_keypoints, :num_keypoints]

        # pad to num of keypoints
        if dist_score_subset.shape[0] < self.qp_max_input_keypoints:
            pad_size = self.qp_max_input_keypoints - dist_score_subset.shape[0]
            dist_score_subset = torch.cat((
                dist_score_subset,
                torch.zeros(pad_size, dtype=dist_score.dtype, device=dist_score.device)
            ))
            rbf_kernel_tmp = torch.zeros(
                self.qp_max_input_keypoints, self.qp_max_input_keypoints, 3,
                dtype=rbf_kernel_subset.dtype, device=rbf_kernel_subset.device
            )
            rbf_kernel_tmp[:num_keypoints, :num_keypoints] = rbf_kernel_subset
            rbf_kernel_subset = rbf_kernel_tmp
        else:
            pad_size = 0

        rbf_kernel_sqrt, _ = sqrt_newton_schulz_autograd(
            rbf_kernel_subset.unsqueeze(0),
            numIters=self.qp_sqrtm_iters, dtype=rbf_kernel_subset.dtype
        )

        try:

            solution, = self.qp_layer(
                rbf_kernel_sqrt.squeeze(0), dist_score_subset.unsqueeze(-1),
                solver_args={"max_iters": self.qp_solver_max_iters}
            )

        except diffcp.cone_program.SolverError:
            print(rbf_kernel_sqrt.shape)


        # clip
        solution = solution.squeeze(0)[:dist_score.shape[0]]

        # pad
        if solution.shape[0] < dist_score.shape[0]:
            pad_size = dist_score.shape[0] - solution.shape[0]
            solution = torch.cat((
                solution,
                torch.zeros(pad_size, dtype=solution.dtype, device=solution.device)
            ))

        return solution
