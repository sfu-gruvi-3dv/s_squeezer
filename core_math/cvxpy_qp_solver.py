import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import diffcp
import torch
from einops import asnumpy

def solve_qp_np(K, d, compression_ratio=0.7, lambda_=1.0):
    # K to make problem DPP https://www.cvxpy.org/tutorial/advanced/index.html
    # K_data = np.random.randn(num_features, num_features)
    num_features = K.shape[0]
    if isinstance(K, torch.Tensor):
        K = asnumpy(K)
    if isinstance(d, torch.Tensor):
        d = asnumpy(d)

    alpha = cp.Variable((num_features, 1))
    constraints = [
        cp.sum(alpha) == 1,
        alpha >= 0,
        alpha <= 1.0 / (compression_ratio * num_features)
    ]

    # this expression non DPP

    objective = cp.Minimize(cp.quad_form(alpha, K) - lambda_ * d.T @ alpha)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="OSQP")

    alpha_ = alpha.value
    return alpha_

class CVXPY_QP:
    def __init__(self, num_features, compression_ratio=0.1):
        self.num_features = num_features
        self.compression_ratio = compression_ratio
        self.K_placeholder = cp.Parameter((num_features, num_features), PSD=True)
        # self.K_sqrt_placeholder = cp.Parameter((num_features, num_features))
        self.d_placeholder = cp.Parameter((num_features, 1))

        self.alpha = cp.Variable((num_features, 1), pos=True)
        constraints = [
            cp.sum(self.alpha) == 1,
            self.alpha >= 0,
            self.alpha <= 1.0 / (compression_ratio * num_features)
        ]
        objective = cp.Minimize(
            # this expression non DPP
            cp.quad_form(self.alpha, self.K_placeholder) - (self.d_placeholder.T @ self.alpha)
            # cp.sum_squares(self.K_sqrt_placeholder @ self.alpha) - (self.d_placeholder.T @ self.alpha)
        )

        self.problem = cp.Problem(objective, constraints)

    def __call__(self, K: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        device = d.device
        dtype = d.dtype
        K = asnumpy(K[:self.num_features, :self.num_features])
        d = asnumpy(d[:self.num_features].view(-1, 1))

        self.d_placeholder.value = d
        self.K_placeholder.value = K

        # K_sqrt = sqrtm(K)
        # self.K_sqrt_placeholder.value = K_sqrt

        self.problem.solve(solver='OSQP')

        return (torch.from_numpy(self.alpha.value).to(device).type(dtype).squeeze(-1), )

    def forward(self, rbf_kernel: torch.Tensor, dist_score: torch.Tensor) -> torch.Tensor:
        return self(rbf_kernel, dist_score)
