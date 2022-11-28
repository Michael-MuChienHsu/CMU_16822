import numpy as np
from utils import *

def get_F_ransac(p1, p2, num_iter=10000, error_threshold = 8, point_num=8):
    """Use ransac to get F, with eightpoint or seven point algorithm.

    Args: 
        pts1: Nx2 points in image 1.
        pts2: Nx2 points in image 2.
        num_iter: times in ransac to iterate.
        error_threshold: threshold to admit a point is inlier.

    Returns:
        F: fundamental matrix.
    """
    assert point_num == 8

    total_matches = p1.shape[0]
    best_inlier_num = 0 
    
    for _ in range(num_iter):
        sample_index = np.random.choice( total_matches, point_num, replace=False)
        _p1, _p2 = p1[sample_index], p2[sample_index]

        if point_num == 8:
            temp_F = eightpoint_F(_p1, _p2)        
        # else:
            # temp_F_list = sevenpoint_F(_p1, _p2)
            # temp_F = pick_sevenpoint_F(temp_F_list, _p1, _p2 )
        
        error = get_epi_error(p1, p2, temp_F)
        inlier_num =  np.sum( error < error_threshold  )

        if inlier_num > best_inlier_num:
            best_F = temp_F
            best_inlier_num = inlier_num
                 
    return best_F 

def recover_R_t_from_E(E):
    """Recover R, t from E.

    Args:
        E: Essentional Matrix.
    
    Returns:
        R: Rotation matrix with det(R) = 1.
        t: translation with sign ambiguity.
    """

    U, S, VT = np.linalg.svd(E)
    k = np.mean(S[:2])
    S = np.array([k, k, 0])
    U, S, VT = np.linalg.svd(U@np.diag(S)@VT)

    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    
    t = U[:,2].reshape([-1, 1])

    if np.linalg.det(U@W@VT)<0:
        W = -W

    R1 = U@W.T@VT
    R2 = U@W@VT

    return  [ np.hstack( [R1, t] ), np.hstack( [R1, -t] ), np.hstack( [R2, t] ), np.hstack( [R2, -t] )]

def get_P1_P2( pt1, pt2, K1, K2, E):
    """Translation t is sign ambiguous. Find t s.t. trangulation gives positive z coordinate.

    Args:
        pt1: 2d homnogeneous points on frame 1. 
        pt2: 2d homnogeneous points on frame 2.
        K1: intrinsic matrix for camera 1.
        K2: intrinsic matrix for camera 2.
        E: Essential matrix.

    Returns:
        P1: Camera1 parameter without ambiguity. 
        P2: Camera2 parameter without ambiguity. 
        3d_point: 3D points after triangulation using P1 and P2.
    """
    P1 = np.hstack( [K1, np.array([ 0, 0, 0 ]).reshape(3, 1) ] )
    extrinsic = recover_R_t_from_E(E)

    max_pos_z = 0
    for _extrinsic in extrinsic:
        _P2 = np.dot( K2, _extrinsic )
        _3d_pts = triangulate( P1, _P2, to_homo(pt1), to_homo(pt2) )
        pos_z = (_3d_pts[:, 2]>=0).sum()
        if pos_z > max_pos_z:
            max_pos_z = pos_z
            R = _extrinsic[:, :3]
            t = _extrinsic[:, -2]
            P2 = _P2
            estimated_3d_points = _3d_pts

    return R, t, P1, P2, estimated_3d_points