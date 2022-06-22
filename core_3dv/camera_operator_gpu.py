import torch


""" Camera Utilities ---------------------------------------------------------------------------------------------------
"""


def camera_center_from_Tcw(Rcw: torch.Tensor, tcw: torch.Tensor) -> torch.Tensor:
    """
        [Batch operations]
        Compute the camera center from extrinsic matrix (world -> camera)

    Args:
        Rcw (tensor): Rotation matrix , world -> camera, dim (N, 3, 3) or (3, 3)
        tcw (tensor): translation of the camera, world -> camera, dim  (N, 3) or (3, )

    Returns:
         camera center in world coordinate system, dim (N, 3)

    """

    keep_dim_n = False
    if Rcw.dim() == 2:
        Rcw = Rcw.unsqueeze(0)
        tcw = tcw.unsqueeze(0)
    N = Rcw.shape[0]
    Rwc = torch.transpose(Rcw, 1, 2)
    C = -torch.bmm(Rwc, tcw.view(N, 3, 1))
    C = C.view(N, 3)

    if keep_dim_n:
        C = C.squeeze(0)
    return C


def translation_from_center(R: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
        [Batch operations]
        Convert center to translation vector, C = -R^T * t -> t = -RC

    Args:
        R (array): Rotation matrix, dim (N, 3, 3) or (3, 3)
        C (array): center of the camera, , dim  (N, 3) or (3, )

    Returns:
         translation vector, dim (N, 3) or (3, )

    """
    keep_dim_n = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        C = C.unsqueeze(0)
    N = R.shape[0]
    t = -torch.bmm(R, C.view(N, 3, 1))
    t = t.view(N, 3)

    if keep_dim_n:
        t = t.squeeze(0)
    return t


def camera_pose_inv(R: torch.Tensor, t: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
        [Batch operations]
        Compute the inverse pose.

    Args:
        R (tensor): Rotation matrix , dim (N, 3, 3) or (3, 3)
        t (tensor): translation of the camera, dim  (N, 3) or (3, )

    Returns:
         Camera pose matrix of (3x4)

    """
    keep_dim_n = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

    N = R.size(0)
    Rwc = torch.transpose(R, 1, 2)
    tw = -torch.bmm(Rwc, t.view(N, 3, 1))

    if keep_dim_n:
        Rwc = Rwc.squeeze(0)
        tw = tw.squeeze(0)

    return Rwc, tw


def transpose(R: torch.Tensor, t: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
        [Batch operations]
        Transform the 3D points by the rotation and the translation.

    Args:
        R (tensor): Rotation matrix , dim (N, 3, 3) or (3, 3)
        t (tensor): translation of the camera, dim  (N, 3) or (3, )
        X (array): points with 3D position, a 2D array with dim of (N, num_points, 3) or (num_points, 3)

    Returns:
        transformed 3D points with dimension of (N, num_points, 3) or (num_points, 3).

    """
    keep_dim_n = False
    keep_dim_hw = False
    H, W = 0, 0
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    if X.dim() == 2:
        X = X.unsqueeze(0)

    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    N = R.shape[0]
    M = X.shape[1]
    X_after_R = torch.bmm(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)

    if keep_dim_hw:
        trans_X = trans_X.view(N, H, W, 3)
    if keep_dim_n:
        trans_X = trans_X.squeeze(0)

    return trans_X


def transform_mat44(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
        [Batch operations]
        Concatenate the 3x4 mat [R, t] to 4x4 mat [[R, t], [0, 0, 0, 1]].

    Args:
        R (tensor): Rotation matrix , dim (N, 3, 3) or (3, 3)
        t (tensor): translation of the camera, dim  (N, 3) or (3, )

    Returns:
        transformation matrix with dim (N, 4, 4) or (4, 4)

    """
    keep_dim_n = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)

    N = R.shape[0]
    bot = torch.tensor([0, 0, 0, 1], dtype=torch.float).to(R.device).view((1, 1, 4)).expand(N, 1, 4)
    b = torch.cat([R, t.view(N, 3, 1)], dim=2)
    out_mat44 = torch.cat([b, bot], dim=1)
    if keep_dim_n:
        out_mat44 = out_mat44.squeeze(0)

    return out_mat44


def Rt(T: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
        Return the rotation matrix and the translation vector.

    Args:
        T (tensor): Transform matrix with dim (N, 3, 4) or (3, 4)

    Returns:
        R (tensor): Rotation matrix , dim (N, 3, 3) or (3, 3)
        t (tensor): translation of the camera, dim  (N, 3) or (3, )

    """

    if T.dim() == 2:
        return T[:3, :3], T[:3, 3]
    elif T.dim() == 3:
        return T[:, :3, :3], T[:, :3, 3]
    else:
        raise Exception("The dim of input T should be either (N, 3, 3) or (3, 3)")


def relative_pose(R_A: torch.Tensor, t_A: torch.Tensor, R_B: torch.Tensor, t_B: torch.Tensor):
    """
        Compute the relative pose from A to B.

    Args:
        R_A (tensor): frame A rotation matrix, dim (N, 3, 3)
        t_A (tensor): frame A translation vector, dim (N, 3) or (3, )
        R_B (tensor): frame B rotation matrix, dim (N, 3, 3)
        t_B (tensor): frame B translation vector, dim (N, 3) or (3, )

    Returns:
         3x3 rotation matrix, 3x1 translation vector that build a 3x4 matrix of T = [R,t]

    """
    keep_dim_n = False
    if R_A.dim() == 2 and t_A.dim() == 2:
        keep_dim_n = True
        R_A = R_A.unsqueeze(0)
        t_A = t_A.unsqueeze(0)
    if R_B.dim() == 2 and t_B.dim() == 2:
        R_B = R_B.unsqueeze(0)
        t_B = t_B.unsqueeze(0)

    A_Tcw = transform_mat44(R_A, t_A)
    A_Twc = batched_mat_inv(A_Tcw)
    B_Tcw = transform_mat44(R_B, t_B)

    # Transformation from A to B
    T_AB = torch.bmm(B_Tcw, A_Twc)
    T_AB = T_AB[:, :3, :]

    if keep_dim_n is True:
        T_AB = T_AB.squeeze(0)

    return T_AB



def transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
        transform points by a transformation matrix.

    Args:
        T (tensor): transformation matrix, dim (N, 3, 4)
        points (tensor): 3D points, dim (N, V, 3)

    Returns:
         N X V x 3 transformed points

    """
    T = to_transformation_matrix(T)
    points_homog = homogeneous_points(points)
    return (T @ points_homog.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]


def homogeneous_points(points: torch.Tensor) -> torch.Tensor:
    """
    :param points: (B, N, 3)
    :return (B, N, 4)
    """
    return torch.cat(
        (points, torch.ones_like(points[..., :1])), dim=-1
    )


def to_transformation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Bx3x4 matrix to Bx4x4"""
    B = matrix.shape[0]
    bottom_row = torch.zeros((B, 1, 4), dtype=matrix.dtype, device=matrix.device)
    bottom_row[:, 0, 3] = 1
    return torch.cat((matrix, bottom_row), dim=1)


""" Projections --------------------------------------------------------------------------------------------------------
"""


def pi(K: torch.Tensor, X: torch.Tensor, eps=1e-5) -> [torch.Tensor, torch.Tensor]:
    """
        Project the X in camera coordinates to the image plane

    Args:
        K (tensor): camera intrinsic matrix, dim (N, 3, 3)
        X (tensor): point position in 3D camera coordinates system, is a 2D array with dimension of (N, num_points, 3)

    Returns:
        Projected 2D pixel position with dim (N, num_points, 2) and the depth for each point with dim (N, 1)

    """
    keep_dim_n = False
    keep_dim_hw = False
    H, W = 0, 0
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if X.dim() == 2:
        X = X.unsqueeze(0)      # make dim (1, num_points, 3)
    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    assert K.size(0) == X.size(0)
    N = K.shape[0]

    X_x = X[:, :, 0:1]
    X_y = X[:, :, 1:2]
    X_z = X[:, :, 2:3] + eps

    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    u_x = (fx * X_x + cx*X_z) / X_z
    u_y = (fy * X_y + cy*X_z) / X_z
    u = torch.cat([u_x, u_y], dim=-1)
    d = X_z

    if keep_dim_hw:
        u = u.view(N, H, W, 2)
        d = d.view(N, H, W)
    if keep_dim_n:
        u = u.squeeze(0)
        d = d.squeeze(0)

    return u, d


def reproject(Rt_cw: torch.Tensor, K: torch.Tensor, X: torch.Tensor):
    """
        Re-project 3D world points `X` to camera represented by intrinsic K and extrinsic Rt

    Args:
        Rt_cw (tensor): camera extrinsic matrix (world to camera), dim (N, 3, 4) or (3, 4)
        K (tensor): camera intrinsic matrix, dim (N, 3, 3) or (3, 3)
        X (tensor): point position in 3D camera coordinates system, is a 2D array with dimension of (N, num_points, 3)

    Returns:
        Projected 2D pixel position with dim (N, num_points, 2) and the depth for each point with dim (N, 1)

    """
    keep_dim = False

    if Rt_cw.dim() == 2:
        Rt_cw = Rt_cw.unsqueeze(0)
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if X.dim() == 2:
        X = X.unsqueeze(0)
        keep_dim = True

    X_local_3d = transpose(Rt_cw[:, :3, :3], Rt_cw[:, :3, 3], X[:, :, :3])
    x_local_2d, depth = pi(K, X_local_3d)
    if keep_dim:
        x_local_2d = x_local_2d.squeeze(0)
        depth = depth.squeeze(0)
    return x_local_2d, depth


def is_in_t(uv: torch.Tensor, depth: torch.Tensor, dim_hw):
    """ Check the point is in the image plane
    """
    x = torch.logical_and(uv[:, 0] > 0, uv[:, 0] < dim_hw[1])
    y = torch.logical_and(uv[:, 1] > 0, uv[:, 1] < dim_hw[0])
    z = torch.logical_and(x, y)
    return torch.logical_and(z, depth.view(-1) > 0)


def pi_inv(K: torch.Tensor, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
        Project the pixel in 2D image plane and the depth to the 3D point in camera coordinate.

    Args:
        K (tensor): camera intrinsic matrix, dim (N, 3, 3)
        x (tensor): 2d pixel position, a 2D array with dim (N, num_points, 2)
        d (tensor): depth at that pixel, a array with dim (N, num_points, 1)

    Returns:
        3D point in camera coordinate, dim (num_points, 3)

    """
    keep_dim_n = False
    keep_dim_hw = False
    H, W = 0, 0
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if x.dim() == 2:
        x = x.unsqueeze(0)      # make dim (1, num_points, 3)
    if d.dim() == 2:
        d = d.unsqueeze(0)      # make dim (1, num_points, 1)

    if x.dim() == 4:
        assert x.size(0) == d.size(0)
        assert x.size(1) == d.size(1)
        assert x.size(2) == d.size(2)
        assert x.size(3) == 2
        keep_dim_hw = True
        N, H, W = x.shape[:3]
        x = x.view(N, H*W, 2)
        d = d.view(N, H*W, 1)

    N = K.shape[0]
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X_x = d * (x[:, :, 0:1] - cx) / fx
    X_y = d * (x[:, :, 1:2] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=-1)

    if keep_dim_hw:
        X = X.view(N, H, W, 3)
    if keep_dim_n:
        X = X.squeeze(0)

    return X


def batched_mat_inv(mat: torch.Tensor) -> torch.Tensor:
    """
        Returns the inverses of a batch of square matrices.

    Args:
        mat (tensor): Batched Square Matrix tensor with dim (N, m, m). N is the batch size, m is the size of square mat.

    Returns:
        inv. matrix
    """
    n = mat.size(-1)
    flat_bmat_inv = torch.stack([m.inverse() for m in mat.view(-1, n, n)])
    return flat_bmat_inv.view(mat.shape)


def x_2d_normalize(h, w, x_2d):
    """
        Convert the x_2d coordinates to (-1, 1)

    Args:
        x_2d: coordinates mapping, (N, H * W, 2)
    
    Returns:
        x_2d: coordinates mapping, (N, H * W, 2), with the range from (-1, 1)

    """
    x_2d[:, :, 0] = (x_2d[:, :, 0] / (float(w) - 1.0))
    x_2d[:, :, 1] = (x_2d[:, :, 1] / (float(h) - 1.0))
    x_2d = x_2d * 2.0 - 1.0
    return x_2d
