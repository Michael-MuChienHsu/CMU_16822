import numpy as np
from q1_utils import *


def get_F_ransac(p1, p2, num_iter=10000, error_threshold = 8, point_num=8):
    """Use ransac to get F, with eightpoint or seven point algorithm.
    1. Random select 8 or 7 (locs1, locs2) pair.
    2. Calculate temp_F.
    3. Count inlier error < error_threshold.
    4. Repeat 1 to 3 num_iter times to find the set of 8 pairs that gives most inlier.
    5. Use found 8 point pair to compute F.

    Args: 
        pts1: Nx2 points in image 1.
        pts2: Nx2 points in image 2.
        num_iter: times in ransac to iterate.
        error_threshold: threshold to admit a point is inlier.

    Returns:
        F: fundamental matrix.
        best_fit_p1: point that gives F with most inliers.
        best_inlier_num_record: list with length = len(iter), best inlier num across iterations.
    """
    assert point_num == 8 or point_num == 7

    total_matches = p1.shape[0]
    best_inlier_num = 0 
    best_inlier_num_record = []
    
    for _ in range(num_iter):
        sample_index = np.random.choice( total_matches, point_num, replace=False)
        _p1, _p2 = p1[sample_index], p2[sample_index]

        if point_num == 8:
            temp_F = eightpoint_F(_p1, _p2)        
        else:
            temp_F_list = sevenpoint_F(_p1, _p2)
            temp_F = pick_sevenpoint_F(temp_F_list, _p1, _p2 )
        
        error = get_epi_error(p1, p2, temp_F)
        inlier_num =  np.sum( error < error_threshold  )

        if inlier_num > best_inlier_num:
            best_F = temp_F
            best_fit_p1, best_fit_p2 = _p1, _p2
            best_inlier_num = inlier_num
        best_inlier_num_record.append(best_inlier_num/total_matches)
                 
    return best_F, best_fit_p1, best_inlier_num_record