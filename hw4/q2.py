"""Incremental Structure-from-Motion"""
from utils import *
import numpy as np
from q2_utils import *

def add_cam( P_list, cor_path_list, _3d_points ):
    """Add new camera with new correspondences to add new 3D point and estimate new camera's param.

    Args:
        P_list: List of knowm camera params.
        cor_path_list: List of path of correspondences betwen P_list and new cameta to estimate.
        _3d_points: Known 3D points. 
    
    Return:
        _3d_points: pPints in 3D, include triangulated new points.
        new_cam_P: Estimated new camera param.
    """
    assert len(P_list) == len(cor_path_list)

    # Place holder for PnP and add new 3D points.
    matched_2d_points = np.empty((0, 2))
    matched_3d_points = np.empty((0, 3))
    correspond_pair_list = []
    new_idx_list = []
    
    # for each known P and correspondes, use known P to regiser 3D points and new camera's 2P point pair
    for _P, _cor_path in zip(P_list, cor_path_list):
        cam1_pts, cam2_pts = read_correp_from_dir(_cor_path)
        _matched_3d_points, matched_idx, new_idx = match_3d_2d_points(cam1_pts, _P, _3d_points)

        matched_2d_points = np.vstack([matched_2d_points, cam2_pts[matched_idx]])
        matched_3d_points = np.vstack([matched_3d_points, _matched_3d_points])
        
        # For triangulation and adding new points to 3D point cloud. 
        correspond_pair_list.append( [cam1_pts, cam2_pts] )
        new_idx_list.append(new_idx)

    # Solve PnP use all matched points
    new_cam_P = PnP( matched_2d_points, matched_3d_points )

    # Use new camera parameter to triangulate new points. Using only new points to solve triangulation is unstable.
    for _P, _new_idx, (cam1_pts, cam2_pts) in zip(P_list, new_idx_list, correspond_pair_list):
        _new_3d_points = triangulate_with_filter(_P, new_cam_P, cam1_pts, cam2_pts, _new_idx)
        _3d_points = np.vstack( [_3d_points, _new_3d_points] )

    return _3d_points, new_cam_P

def initalize_sfm(P1, P2, pair_path):
    """Initialze 3D points for incremental SfM.

    Args:
        P1: Camera param.
        P2: Camera param.
        pair_1_2_path: Path to correspondence pair.

    Return:
        _3d_points: Estimated 3D points.
    """
    cam1_pts, cam2_pts = read_correp_from_dir(pair_path)
    _3d_points = triangulate(P1, P2, to_homo(cam1_pts), to_homo(cam2_pts))
    return _3d_points

def q2_main(P1, P2):
    """Incremental SfM. Use P1 P2 to initialize, and add camera 3 and camera 4 sequentially."""
    K1, R1, t1 = P1["K"], P1["R"], P1["T"]
    K2, R2, t2 = P2["K"], P2["R"], P2["T"]
    P1 = K1@(np.hstack( [R1, t1.reshape(3, 1)] ))
    P2 = K2@(np.hstack( [R2, t2.reshape(3, 1)] ))
    _K = K1    
    
    _3d_points =  initalize_sfm(P1, P2, pair_1_2_path)
    print("Initialize 3D points with P1, P2")
    plot_3d(_3d_points)
    
    _3d_points, P3 = add_cam( [P1, P2], [pair_1_3_path, pair_2_3_path], _3d_points )
    plot_3d(_3d_points)
    R3, t3 = get_R_t( P3, _K )
    print(f"Decompose P3\nR3\n{R3}]\nt3: {t3}")

    _3d_points, P4 = add_cam( [P1, P2, P3], [pair_1_4_path, pair_2_4_path, pair_3_4_path], _3d_points)
    plot_3d(_3d_points)
    R4, t4 = get_R_t( P4, _K )
    print(f"Decompose P4\nR4\n{R4}]\nt4: {t4}")

    return None

if __name__ == "__main__":
    P1 = np.load("./data/data_cow/cameras/cam1.npz")
    P2 = np.load("./data/data_cow/cameras/cam2.npz")

    pair_1_2_path = "./data/data_cow/correspondences/pairs_1_2/"
    pair_1_3_path = "./data/data_cow/correspondences/pairs_1_3/"
    pair_2_3_path = "./data/data_cow/correspondences/pairs_2_3/"
    pair_1_4_path = "./data/data_cow/correspondences/pairs_1_4/"
    pair_2_4_path = "./data/data_cow/correspondences/pairs_2_4/"
    pair_3_4_path = "./data/data_cow/correspondences/pairs_3_4/"

    q2_main(P1, P2)