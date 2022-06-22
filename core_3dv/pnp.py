import pycolmap
import numpy as np
import core_math.transfom as mtrans

def estimate(point2d: np.ndarray, point3d: np.ndarray, cam_K: np.ndarray, dim_hw: np.ndarray or list, reproj_thres=4.0):

    assert point2d.ndim == 2 and point2d.shape[1] == 2
    assert point3d.ndim == 2 and point3d.shape[1] == 3
    assert cam_K.ndim == 2

    ret = pycolmap.absolute_pose_estimation(point2d.astype(np.float32), 
                                            point3d.astype(np.float32),
                                            {
                                                'model': 'SIMPLE_PINHOLE',
                                                'width': int(dim_hw[1]),
                                                'height': int(dim_hw[0]),
                                                'params': [cam_K[0, 0], cam_K[0, 2], cam_K[1, 2]]
                                            }, 
                                            reproj_thres)

    if ret['success'] == True:
        Tcw = np.eye(4)
        Tcw[:3, :3] = mtrans.quaternion_matrix(ret['qvec'])[:3, :3]
        Tcw[:3, 3] = ret['tvec']

        inliers = np.where(np.asarray(ret['inliers']) == True)[0]
        ret = dict()
        ret['success'] = True
        ret['Tcw'] = Tcw.astype(np.float32)
        ret['inliers'] = inliers
        
    return ret