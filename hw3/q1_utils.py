import numpy as np
import cv2 as cv

COLOR_LIST = [(60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
              (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), 
              (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
              (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]

def enforce_singularity(F):
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    return U.dot( np.diag(S).dot(VT) )

def get_normalize_mat_T(points):
    """Compute normalize matrix for numirical stability."""
    points = points[:, :2]
    p_avg = np.mean(points, axis=0)
    x0, y0 = p_avg[0], p_avg[1]
    d_avg = np.mean( np.linalg.norm((points-p_avg)[:, :2], axis=1) )
    s = (np.sqrt(2))/d_avg
    T = np.array( [ [s, 0, -s*x0],
                    [0, s, -s*y0],
                    [0, 0, 1] ] )
    return T

def to_homo(points):
    """Conver points to homogenous point.
    Args:
        points: Nx2 np.adarray points
    
    Returns:
        homo_points: Nx3 points    
    """
    if points.shape[1] == 3:
        return points
    return np.hstack( [points, np.ones( (points.shape[0], 1) )] )

def eightpoint_F(p1, p2):
    """Get F using eight point algorithm.

    Args:
        p1: Nx2 or Nx3 ndarray points.
        p2: : Nx2 or Nx3 ndarray points
    
    Returns:
        F: 3x3 fundamental matrix.
    """

    p1, p2 = to_homo(p1), to_homo(p2)
    T1, T2 = get_normalize_mat_T(p1), get_normalize_mat_T(p2)

    # Normalize
    # p1_hat, p2_hat = p1.dot(T1), p2.dot(T2)
    p1_hat, p2_hat = T1.dot(p1.T).T, T2.dot(p2.T).T

    # Neat
    A = np.matmul(p2_hat[..., None], p1_hat[:, None, :]) 
    A = A.reshape((-1,9))

    # x_, y_ = p1_hat[:, 0], p1_hat[:, 1]
    # x_prime, y_prime = p2_hat[:, 0], p2_hat[:, 1]
    # A = np.vstack( ( x_*x_prime, x_prime*y_, x_prime, y_prime*x_, y_*y_prime, y_prime, x_, y_, np.ones_like(x_prime) ) ).T

    _, _, VT = np.linalg.svd(A)
    F_hat = VT[ -1 ].reshape((3, 3))
    F_hat = enforce_singularity(F_hat)

    # Un-normalize
    F = (T2.T).dot(F_hat).dot(T1)
    F = F/F[2,2]
    
    return F

def get_E_from_F(K1, K2, F):
    """Calculate E given intrinsics and F"""
    E = K2.T@F@K1 
    return E/E[-1][-1]

def get_epipolar_line_y_end_points(F, x, x_max):
    l = F.dot(x)
    l = l/l[-1]
    a, b, c = l
    start_y = int( -c/b )
    end_y = int( (-c-a*x_max)/b )
    # print( np.array([0, start_y, 1]).dot(l) )
    # print( np.array([x_max, end_y, 1]).dot(l) )

    return start_y, end_y

def show_epipolar_line(im1, im2, F, x, show=False):
    """Show epipolar line of a point on im1 to im2"""
    y_max, x_max, _ = im2.shape
    if len(x.shape) == 1:
        start_y, end_y = get_epipolar_line_y_end_points(F, x, x_max)
        cv.circle(im1, x[:2].astype(int), 4, (0, 0, 255), -1)
        cv.line(im2, (0, start_y), (x_max, end_y), (0, 0, 255), 4) 
    else:
        for i, _x in enumerate(x):  
            start_y, end_y = get_epipolar_line_y_end_points(F, _x, x_max)
            cv.circle(im1, _x[:2].astype(int), 12, COLOR_LIST[i%len(COLOR_LIST)], -1)
            cv.line(im2, (0, start_y), (x_max, end_y), COLOR_LIST[i%len(COLOR_LIST)], 8) 
            # print( start_y-end_y )

    if show:
        cv.imshow("im1", im1)
        cv.imshow("im2", im2)
        cv.waitKey(0)

    return im1, im2


def sevenpoint_F(p1, p2):
    """Calculate F using seven point algorithm.

    Args:
        p1: 7 points
        p2: 7 points
    Returns:
        all_possible_F: list of Fs. Sevenpoint algorithm can leads to 1 or 3 solutions.    
    """

    p1, p2 = to_homo(p1), to_homo(p2)
    T1, T2 = get_normalize_mat_T(p1), get_normalize_mat_T(p2)

    # Normalize
    p1_hat, p2_hat = T1.dot(p1.T).T, T2.dot(p2.T).T

    # Neat
    A = np.matmul(p2_hat[..., None], p1_hat[:, None, :]) 
    A = A.reshape((-1,9))

    _, _, VT = np.linalg.svd(A)

    f1 = VT[ -1 ].reshape((3, 3))
    f2 = VT[ -2 ].reshape((3, 3))

    # F = a*f1 + (1-a)*f2 that leads to det(F) = 0
    combination_n1 =  -f1 + 2*f2    # a = -1
    combination_0  =  f2            # a =  0
    combination_p1 =  f1            # a =  1
    combination_p2 =  2*f1 - f2     # a =  2

    a = np.array( [ [1,-1, 1,-1],
                    [1, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 2, 4, 8] ] )

    b = np.array( [ np.linalg.det( combination_n1), 
                    np.linalg.det( combination_0 ),
                    np.linalg.det( combination_p1 ),
                    np.linalg.det( combination_p2 ) ] )

    roots = np.polynomial.polynomial.polyroots( np.linalg.solve( a, b ) )
    real_roots = np.real( roots[ np.isreal(roots) ] )     

    all_possible_F = []
    for root in real_roots:
        F_hat = root*f1 + (1-root)*f2
        F_hat = enforce_singularity(F_hat)

        # un normalize
        F = (T2.T).dot(F_hat).dot(T1)
        F = F/F[2,2]
        all_possible_F.append(F)

    return all_possible_F

def pick_sevenpoint_F( F_list, p1, p2 ):
    """Given a list of F, return the F that has least epipolar error.
    
    Args:
        F_list: list of Fs
        p1: N points
        p2: N points
    
    Returns:
        best_F: F in F_list that has smallest epipolar error.
    """
    p1, p2 = to_homo(p1), to_homo(p2)
    error = np.inf
    for _F in F_list:
        error_temp = np.mean(get_epi_error( p1, p2, _F ))
        if error_temp < error:
            best_F = _F
            error = error_temp

    return best_F

def get_epi_error(p1, p2, F):
    """Calculate distandce between p1, p2 with its corresponding epipolar line. 
    I have looked at my previous 16-720 hw5 for calculating epipolar error.  

    Args: 
        p1: Nx3 array
        p2: Nx3 array
    
    Returns:
        error: Nx1 array
    """
    line1s = p1.dot(F.T)
    line1s = line1s/line1s[:, -1, None]
    dist1 = np.square(np.divide(np.sum(np.multiply(line1s, p2), axis=1), 
                                np.linalg.norm(line1s[:, :2], axis=1)))

    line2s = p2.dot(F)
    line2s = line2s/line2s[:, -1, None]
    dist2 = np.square(np.divide(np.sum(np.multiply(line2s, p1), axis=1),
                                np.linalg.norm(line2s[:, :2], axis=1)))

    error = dist1 + dist2
    return error