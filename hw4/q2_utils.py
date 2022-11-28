# from utils import *
from q1_utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
# from scipy.linalg import expm

_REPRO_THRESHOLD_px = 1

def plot_3d(estimated_3d_points):
    """Plot points in 3d with matplotlib.

    Args:
        estimated_3d_points: Nx3 euclid points in 3d.
    
    Return:
        None.
    """
    cmap = plt.cm.get_cmap("jet")
    X, Y, Z = estimated_3d_points.T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z ,c = -(Z-min(Z)), cmap=cmap, s=2)
    plt.show()

def read_correp_from_dir(dir_path):
    """Read corespondences from 2 view in a directory."""
    npy_path = sorted([ os.path.join(dir_path, _dir) for _dir in os.listdir(dir_path) if _dir[-4:] == ".npy" ])
    c1, c2 = np.load(npy_path[0]), np.load(npy_path[1])
    return c1, c2

def get_R_t(P, K):
    """Decompose P given K for R and t.

    Args:
        P: Camera Parameter.
        K: Intrinsic matrix.

    Returns:
        R: Rotation matrix.
        t: Translation vector.    
    """
    # Solution 1, Cannot guarentee R is SE3.
    # P = K[R|t]
    # R_t = np.linalg.inv(K)@P
    # R, t = R_t[:, :3], R_t[:, -1]    

    # Solution 2
    # P = [KR|Kt] with qr decomposition.
    _, _, VT = np.linalg.svd(P)
    t = (VT[-1]/VT[-1][-1])[:-1]

    M = P[:, :3] # M = KR
    R, _K = np.linalg.qr(M)

    return R, t

def PnP( _2d_pts, _3d_pts ):
    """ PnP to get camera paramter.

    Args: 
        _2d_pts: 2d euclid points.
        _3d_pts: 3d euclid points.
    
    Return:
        P: Camara matix
    """
    x, y = _2d_pts.T # N, N
    _3d_pts = to_homo(_3d_pts, 3) # Nx4
    zeros = np.zeros_like(_3d_pts)

    # DLT
    # [ [ X Y Z W 0 0 0 0 -xX -xY -xZ -xW] 
    #   [ 0 0 0 0 X Y Z W -xX -xY -xZ -xW] ] . dot( P1 P2 P3 ) = 0
    A = np.vstack( [ np.hstack([ _3d_pts, zeros, -x[:, None]*_3d_pts ]), 
                     np.hstack([ zeros, _3d_pts, -y[:, None]*_3d_pts ]) ] )
    _, _, VT = np.linalg.svd( A )
    P = VT[-1].reshape((3, 4))

    return P/P[-1][-1]

def match_3d_2d_points(_2d_pts, P, _3d_points, threshold = _REPRO_THRESHOLD_px):
    """Match 3d points with 2d points with L2 reprojection error.
    
    Args:
        _2d_pts: Euclid points in 2D, Nx2.
        P: Camera Matrix for P.
        _3d_points: Euclid points in 3D, Mx3.
        threshold: Threshold for inlier.

    Returns:
        matched_3d_points: 3d points corresponding to 2D points.
        matched_2d_idx: Matching table for 2d index.
        unmatched_2d_idx: Possible new points.
    """
    _repro_pts = normalize_points(to_homo(_3d_points, 3)@P.T)

    matched_2d_idx = []
    unmatched_2d_idx = []
    matched_3d_points = np.empty((0, 3))
    for i, _pts in enumerate(_2d_pts):
        _3d_idx = np.argmin( np.linalg.norm( _repro_pts-_pts, axis = 1 ) )
        min_dist = min(np.linalg.norm( _repro_pts-_pts, axis = 1 ))
        if min_dist < threshold:
            matched_3d_points = np.vstack( [matched_3d_points, _3d_points[_3d_idx].reshape(1, 3)] )
            matched_2d_idx.append(i)
        else:
            unmatched_2d_idx.append(i)

    return matched_3d_points , matched_2d_idx, unmatched_2d_idx

def triangulate_with_filter(P1, P2, cam1_pts, cam2_pts, new_point_idx):
    """Triangulate and return selected 3d points with reprojection error < _REPRO_THRESHOLD. 
    Since triangulation is solving a DLT with SVD, triangulate only on selected points could be unstable, so we have
    to triangulate with all registered points and return the selected ones.

    Args:
        P1: Camera Param.
        P2: Camera Param.
        cam1_pts: 2d euclid points.
        cam2_pts: 2d euclid points.
        new_point_idx: Mapping table (indices).
    
    Returns:
        _3d_points: Points in 3d.    
    """
    _3d_points = triangulate( P1, P2, to_homo(cam1_pts), to_homo(cam2_pts) ) 
    _3d_points = _3d_points[new_point_idx]
    new_cam2_pts = cam2_pts[new_point_idx]
    new_cam1_pts = cam1_pts[new_point_idx]

    # Discard outliers.
    error = np.linalg.norm( normalize_points(to_homo(_3d_points, 3)@P2.T) - new_cam2_pts, axis = 1 )
    error += np.linalg.norm( normalize_points(to_homo(_3d_points, 3)@P1.T) - new_cam1_pts, axis = 1 )
    _3d_points = _3d_points[ error < 2*_REPRO_THRESHOLD_px ]
    return _3d_points

def bundle_adjustment():
    pass