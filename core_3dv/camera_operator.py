import numpy as np
from numpy.linalg import inv as np_mat_inv


''' Matrices -----------------------------------------------------------------------------------------------------------
'''


def P_mat(K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
        Camera matrix of P = K * [R|t], 3x4 matrix

    Args:
        K (array): Camera intrinsic matrix (3x3)
        T (array): Camera extrinsic matrix (3x4)

    Returns:
         P = K * T, projection matrix (3x4)

    """
    return np.dot(K, T)


def K_from_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    """
        Generate K matrix from intrinsic array

    Args:
        intrinsic (array): array with 4 items (fx, fy, cx, cy)

    Returns:
        intrinsic matrix with 3x3 elements

    """
    return np.asarray([intrinsic[0], 0, intrinsic[2],
                       0, intrinsic[1], intrinsic[3], 0, 0, 1], dtype=np.float32).reshape(3, 3)


''' Camera Information -------------------------------------------------------------------------------------------------
'''


def Rt(T: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
        Return the rotation matrix and the translation vector.

    Args:
        T (array): Transform matrix with dim (3, 4) or (4, 4)

    Returns:
        Rcw (array): Rotation matrix with dimension of (3x3)
        tcw (array): translation vector (3x1)

    """
    return T[:3, :3], T[:3, 3]


def camera_center_from_Tcw(Rcw: np.ndarray, tcw: np.ndarray) -> np.ndarray:
    """
        Compute the camera center from extrinsic matrix (world -> camera)

    Args:
        Rcw (array): Rotation matrix with dimension of (3x3), world -> camera
        tcw (array): translation of the camera, world -> camera

    Returns:
        camera center in world coordinate system.

    """
    # C = -Rcw' * t
    C = -np.dot(Rcw.transpose(), tcw)
    return C


def translation_from_center(R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
        Convert center to translation vector, C = -R^T * t -> t = -RC

    Args:
        R (array): Rotation matrix with dimension of (3x3)
        C (array): center of the camera

    Returns:
        translation vector

    """
    t = -np.dot(R, C)
    return t


def camera_center_from_P(P: np.ndarray) -> np.ndarray:
    """
        Compute the camera center from camera matrix P, where P = [M | -MC], M = KR, C is the center of camera.
        The decompose method can be found in Page 163. (Multi-View Geometry Second Edition)

    Args:
        P (array): Camera projection matrix, dim (3x4)

    Returns:
        camera center C at world coordinate system.

    """
    X = np.linalg.det(np.asarray([P[:, 1],  P[:, 2], P[:, 3]]))
    Y = -np.linalg.det(np.asarray([P[:, 0],  P[:, 2], P[:, 3]]))
    Z = np.linalg.det(np.asarray([P[:, 0],  P[:, 1], P[:, 3]]))
    T = -np.linalg.det(np.asarray([P[:, 0],  P[:, 1], P[:, 2]]))
    C = np.asarray([X, Y, Z]) / T
    return C


def camera_pose_inv(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
        Compute the inverse pose.

    Args:
        R (array): Rotation matrix with dimension of (3x3)
        t (array): translation vector with dim of (3x1)

    Returns:
        Camera pose matrix of (3x4)

    """
    Rwc = R.transpose()
    Ow = - np.dot(Rwc, t)
    Twc = np.eye(4, dtype=np.float32)
    Twc[:3, :3] = Rwc
    Twc[:3, 3] = Ow
    return Twc[:3, :]


def fov(fx: float, fy: float, h: float, w: float) -> [float, float]:
    """
        Camera fov on x and y dimension

    Args:
        fx (float): focal length on x axis
        fy (float): focal length on y axis
        h (float): frame height
        w (float): frame width

    Returns:
        field of view on x and y axis.

    """
    return np.rad2deg(2*np.arctan(w / (2*fx))), np.rad2deg(2*np.arctan(h / (2*fy)))


''' Camera Projection --------------------------------------------------------------------------------------------------
'''


def pi(K: np.ndarray, X: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
        Project the X in camera coordinates to the image plane

    Args:
        K (array): camera intrinsic matrix 3x3
        X (array): point position in 3D camera coordinates system, is a 2D array with dimension of (num_points, 3)

    Returns:
        Projected 2D pixel position with dim (num_points, 2) and the depth for each point.

    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.zeros((X.shape[0], 2), dtype=np.float32)
    u[:, 0] = fx * X[:, 0] / (X[:, 2] + 1e-5) + cx
    u[:, 1] = fy * X[:, 1] / (X[:, 2] + 1e-5) + cy
    return u, X[:, 2]


def pi_inv(K: np.ndarray, x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
        Project the pixel in 2D image plane and the depth to the 3D point in camera coordinate.

    Args:
        K (array): camera intrinsic matrix contains (fx, fy, cx, cy), dim (3x3)
        x (array): 2d pixel position, a 2D array with dimension of (num_points, 2)
        d (array): depth at that pixel, a array with dimension of (num_points, 1)

    Returns:
        3D point in camera coordinate, dim (num_points, 3)

    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = np.zeros((x.shape[0], 3), dtype=np.float32)
    X[:, 0] = d[:, 0] * (x[:, 0] - cx) / fx
    X[:, 1] = d[:, 0] * (x[:, 1] - cy) / fy
    X[:, 2] = d[:, 0]
    return X


''' Camera Transform ---------------------------------------------------------------------------------------------------
'''


def relateive_pose(R_A: np.ndarray, t_A: np.ndarray, R_B: np.ndarray, t_B: np.ndarray) -> np.ndarray:
    """
        Compute the relative pose from A to B.

    Args:
        R_A (array): frame A rotation matrix (3x3)
        t_A (array): frame A translation vector (3x1)
        R_B (array): frame B rotation matrix (3x3)
        t_B (array): frame B translation vector (3x1)

    Returns:
        3x3 rotation matrix, 3x1 translation vector that build a 3x4 matrix of T = [R,t]

    Example:
        >>> C_A = camera_center_from_Tcw(R_A, t_A)
        >>> C_B = camera_center_from_Tcw(R_B, t_B)
        >>> R_AB = np.dot(R_B, R_A.transpose())
        >>> t_AB = np.dot(R_B, C_A - C_B)
    """

    A_Tcw = np.eye(4, dtype=np.float32)
    A_Tcw[:3, :3] = R_A
    A_Tcw[:3, 3] = t_A
    A_Twc = np_mat_inv(A_Tcw)

    B_Tcw = np.eye(4, dtype=np.float32)
    B_Tcw[:3, :3] = R_B
    B_Tcw[:3, 3] = t_B

    # transformation from A to B
    T_AB = np.dot(B_Tcw, A_Twc)
    return T_AB[:3, :]


def transpose(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
        Transform the 3D points by the rotation and the translation.

    Args:
        R (array): rotation matrix in dimension of 3x3
        t (array): translation vector with 3 elements.
        X (array): points with 3D position, a 2D array with dimension of (num_points, 3)

    Returns:
         transformed 3D points with dimension of (num_points, 3)

    """

    assert R.shape[0] == 3
    assert R.shape[1] == 3
    assert t.shape[0] == 3
    trans_X = np.dot(R, X.transpose()).transpose() + t
    return trans_X


def rel_R_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
        rotation angle between two rotations
    Args:
        R1 (array): rotation matrix in dimension of 3x3
        R2 (array): rotation matrix in dimension of 3x3

    Returns:
        the relative rotation angle between R1 and R2.

    """

    # relative pose
    rel_R = np.matmul(R1[:3, :3], R2[:3, :3].T)
    trace = np.clip((np.trace(rel_R) - 1) / 2, a_min=-1, a_max=1)
    R_err = np.rad2deg(np.arccos(trace))
    return R_err
