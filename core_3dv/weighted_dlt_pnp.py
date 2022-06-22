import numpy as np
import torch
# ref: https://github.com/acvictor/DLT/blob/master/DLT.py


def DLT_RT(K, uv, xyz, weight=None, soft_sign=True):
    """
    Computing Projection Matrix by DLT using known object points and their image points.

    [Verified, see dbg_qp_pnploss.ipynb]

    Args:
        K (torch.Tensor): camera intrinsic, dim (3, 3)
        uv (torch.Tensor): 2D keypoint position, dim (M, 2)
        xyz (torch.Tensor): 3D keypoint position, dim (M, 3)
        weight (torch.Tensor): weight for each pair, range (0, 1), dim (M, 3)

    Returns:
        P (torch.Tensor): projection matrix: (3, 4)

    """

    M = xyz.shape[0]
    assert M > 6 and xyz.shape[1] == 3              # DLT requires at least 6 points
    assert uv.shape[0] == M and uv.shape[1] == 2

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # construct matrix A: --------------------------------------------------------------------------------------------------
    h_xyz = torch.cat([xyz, torch.ones((M, 1), device=xyz.device)], dim=1)   # homogeneous coordinate, dim: (M, 4)

    A00 = h_xyz * fx                                   # [ x*f_x, y*f_x, z*f_x, 1*f_x], dim: (M, 4)
    A01 = torch.zeros((M, 4), device=xyz.device)
    A10 = torch.zeros((M, 4), device=xyz.device)
    A11 = h_xyz * fy                                   # [ x*f_y, y*f_y, z*f_y, 1*f_y], dim: (M, 4)

    x, y, z = h_xyz[:, 0], h_xyz[:, 1], h_xyz[:, 2]
    u, v = uv[:, 0], uv[:, 1]

    cx_s_u, cy_s_v = cx - u, cy - v
    B0 = [cx_s_u * x, cx_s_u * y, cx_s_u * z, cx_s_u]
    B0 = torch.cat([b.view(M, 1) for b in B0], dim=1)    # dim: (M, 4)
    B1 = [cy_s_v * x, cy_s_v * y, cy_s_v * z, cy_s_v]
    B1 = torch.cat([b.view(M, 1) for b in B1], dim=1)    # dim: (M, 4)

    A = torch.stack([A00, A01, B0,
                    A10, A11, B1], dim=1).view((M * 2, 12))

    # Solve Least Square AX=0 ------------------------------------------------------------------------------------------
    if weight is not None:
        W = torch.cat([weight.view(M, 1), weight.view(M, 1)], dim=1).view(M * 2).contiguous()
        A = W.view(M * 2, 1) * A

    U, _, Vt = torch.linalg.svd(A)       # note: Vt stands for the transposed of V, see torch documents
    Rt_ = Vt[-1, :]
    Rt_ = Rt_.view(3, 4)

    # Extract real rotation matrix R and make sure its orthodiagonal property
    U, D, V = torch.linalg.svd(Rt_[:3, :3])
    beta = 1 / (torch.sum(D) / 3)                  # beta is a scale factor

    # check if beta * (x*a31 + y*a32 + z*a33 + a34) > 0
    test = torch.sum(beta * h_xyz[0, :].view(4) * Rt_[-1, :].view(4))
    if not soft_sign:
        beta = beta if test > 0 else -beta         # todo: use softsign for fully differentiable version
    else:
        t = 5000.0                                 # control the slope near test=0, the higher the more steep
        beta = torch.nn.functional.softsign(t * test) * beta

    # recover R and t based on the beta
    new_Rt = torch.eye(4)[:3, :].to(A.device)
    D = torch.diag(D * beta)
    new_Rt[:3, :3] = torch.matmul(U, torch.matmul(D, V))
    new_Rt[:3, 3] = Rt_[:3, 3] * beta

    return new_Rt


def normalization_(x: torch.Tensor):
    '''  Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

    Args:
     x: the data to be normalized, dim: (M, d), where M is the number of points and d is the dimension.

    Outputs:
        Tr_inv: the transformation matrix (translation plus scaling)
        x: the transformed (normalized) data

    '''
    M, dim = x.shape
    mean, scale = x.mean(dim=0), x.std()

    Tr = torch.eye(x.shape[1] + 1, device=x.device) * scale
    Tr[-1, -1] = 1.0
    Tr[:dim, -1] = mean

    Tr_inv = torch.linalg.inv(Tr)
    h_x = torch.cat([x, torch.ones((M, 1), device=x.device)], dim=1)            # homogeneous coordinate
    x = torch.matmul(Tr_inv, h_x.transpose(0, 1))
    x = x[:dim, :].transpose(0, 1)

    return Tr_inv, x


def DLT_P(uv, xyz, weight=None):
    """
    Computing Projection Matrix by DLT using known object points and their image points.

    Ref: http://rpg.ifi.uzh.ch/docs/teaching/2020/03_camera_calibration.pdf page 13
    [Verified, see dbg_qp_pnploss.ipynb]

    Args:
        uv (torch.Tensor): 2D keypoint position, dim (M, 2)
        xyz (torch.Tensor): 3D keypoint position, dim (M, 3)
        weight (torch.Tensor): weight for each pair, range (0, 1), dim (M, 1)
        normalize_P (bool): normalize the output P if needed (optional)

    Returns:
        P (torch.Tensor): projection matrix: (3, 4)

    """

    M = xyz.shape[0]
    assert M > 6 and xyz.shape[1] == 3              # DLT requires at least 6 points
    assert uv.shape[0] == M and uv.shape[1] == 2

    Tr_xyz, xyz = normalization_(xyz)
    Tr_uv, uv = normalization_(uv)

    # construct matrix A: ----------------------------------------------------------------------------------------------
    h_xyz = torch.cat([xyz, torch.ones((M, 1), device=xyz.device)], dim=1)   # homogeneous coordinate, dim: (M, 4)

    x, y, z = h_xyz[:, 0], h_xyz[:, 1], h_xyz[:, 2]
    u, v = uv[:, 0], uv[:, 1]
    B0 = [-u * x, -u * y, -u * z, -u]
    B0 = torch.cat([b.view(M, 1) for b in B0], dim=1)    # dim: (M, 4)
    B1 = [-v * x, -v * y, -v * z, -v]
    B1 = torch.cat([b.view(M, 1) for b in B1], dim=1)    # dim: (M, 4)

    A = torch.stack([h_xyz, torch.zeros((M, 4), device=xyz.device), B0,
                     torch.zeros((M, 4), device=xyz.device), h_xyz, B1], dim=1).view((M * 2, 12))

    # Solve Least Square AX=0 ------------------------------------------------------------------------------------------
    if weight is not None:
        W = torch.cat([weight.view(M, 1), weight.view(M, 1)], dim=1).view(M * 2).contiguous()
        A = W.view(M * 2, 1) * A

    _, _, Vt = torch.linalg.svd(A)       # note: Vt stands for the transposed of V, see torch documents
    P = Vt[-1, :] / Vt[-1, -1]
    P = P.view(3, 4)

    # Denormalize ------------------------------------------------------------------------------------------------------
    # ref: https://www.ece.mcmaster.ca/~shirani/vision/hartley_ch7.pdf page 10
    P = torch.matmul(torch.matmul(torch.linalg.pinv(Tr_uv), P), Tr_xyz)
    P = P / P[-1, -1]

    return P.view(3, 4)


if __name__ == '__main__':
    from pathlib import Path
    import random

    from dataset.data_module import CachedDataModule
    from core_dl.expr_ctx import ExprCtx
    import core_3dv.camera_operator_gpu as cam_opt_gpu
    from core_math.transfom import euler_from_matrix

    random.seed(10)
    np.random.seed(10)

    data_model = CachedDataModule.load_from_disk(
        Path('/tmp/squeezer/dbg', 'cached_data.bin')
    )
    loader = data_model.train_dataloader()

    debug = False

    def approx_equal_RT(P1, P2):
        R = torch.matmul(P1[:3, :3], P2[:3, :3].transpose(0, 1))
        r_err = euler_from_matrix(R)
        r_err = np.linalg.norm(np.rad2deg(r_err))

        t = P1[:3, -1] - P2[:3, -1]
        t_err = torch.norm(t, dim=0)

        if debug:
            print('r/t err', r_err, t_err)

        return r_err < 5 and t_err < 0.5

    def zero_pad(x):
        return torch.cat((
            x, torch.ones(x.shape[0], 1)
        ), dim=1)

    for sample in loader:
        fq_imgs, fq_segs, fq_info, rp_imgs, rp_info = sample[:5]
        for img_idx in range(len(rp_imgs)):
            K = rp_info['K'][img_idx]
            Tcw = rp_info['Tcws'][img_idx][0]

            # P_gt = torch.matmul(K[0], Tcw[0])

            # projection matrix is just the extrinsics when using normalized image coords
            P_gt = Tcw.clone()

            P_gt = P_gt / P_gt[-1, -1]

            if debug:
                print('gt P:\n', P_gt, '\n')
                print('gt RT:\n', Tcw, '\n')

            r_pt3d = rp_info['pt3d'][0, rp_info['pt2d_obs3d'][img_idx][0]]

            r_pt2d = rp_info['pt2d_pos'][img_idx][0]
            # normalize 2d coords
            r_pt2d = torch.matmul(
                K[0].inverse(), zero_pad(r_pt2d).transpose(0, 1)
            ).transpose(0, 1)[:, :2]

            rl_gt_pt2d, _ = cam_opt_gpu.reproject(
                Tcw.unsqueeze(0),
                # K, # unnormalized coords
                torch.eye(3).unsqueeze(0), # normalized coords
                r_pt3d
            )

            # pt2d_err = (r_pt2d - rl_gt_pt2d).norm(dim=-1).mean()
            # print(pt2d_err)

            # add outliers
            num_outliers = int(r_pt3d.shape[0] * 0.25)
            outliers = random.sample(range(r_pt3d.shape[0]), num_outliers)
            rl_gt_pt2d_outliers = rl_gt_pt2d.clone()
            for outlier_idx in outliers:
                rl_gt_pt2d_outliers[outlier_idx, 0] += np.random.uniform(-1, 1)
                rl_gt_pt2d_outliers[outlier_idx, 1] += np.random.uniform(-1, 1)

            # no weight
            weight = None
            P_unweighted = DLT_P(rl_gt_pt2d_outliers, r_pt3d, weight)
            RT_unweighted = DLT_RT(torch.eye(3), rl_gt_pt2d_outliers, r_pt3d, weight)
            if debug:
                print('P unweighted: \n', P_unweighted, '\n')
                print('RT unweighted: \n', RT_unweighted, '\n')
            assert(not approx_equal_RT(P_unweighted * Tcw[-1, -1], Tcw))
            assert(not approx_equal_RT(RT_unweighted, Tcw))

            # smart weight
            weight = torch.ones((r_pt3d.shape[0], 1))
            for outlier_idx in outliers:
                weight[outlier_idx] = 1e-7

            P_weighted = DLT_P(rl_gt_pt2d_outliers, r_pt3d, weight)
            RT_weighted = DLT_RT(torch.eye(3), rl_gt_pt2d_outliers, r_pt3d, weight, soft_sign=True)
            if debug:
                print('P weighted: \n', P_weighted, '\n')
                print('RT weighted: \n', RT_weighted, '\n')

            assert(approx_equal_RT(P_weighted * Tcw[-1, -1], Tcw))
            # assert(approx_equal_RT(RT_weighted, Tcw))

            assert(not approx_equal_RT(P_weighted * Tcw[-1, -1], P_unweighted * Tcw[-1, -1]))
            assert(not approx_equal_RT(RT_weighted, RT_unweighted))
