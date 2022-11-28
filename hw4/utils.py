import numpy as np

def normalize_points(pts):
    """Normalize point from homogeneous to euclid.

    Args:
        pts: Nx4 or Nx3 points 

    Returns:
        pts in Euclid space.

    """
    if len(pts.shape) == 1:
        return (pts/pts[-1])[:-1]
    return (pts/pts[:, -1, None])[:, :-1]

def to_skew(pts):
    """Vector to skew symmetric matrix for cross product."""
    x, y, z = pts.T
    x, y, z = x[:, None], y[:, None], z[:, None]
    zeros = np.zeros_like( x )
    skew = np.hstack( [zeros, -z, y,
                       z, zeros, -x,
                       -y, x, zeros ] )
    return skew.reshape(-1, 3, 3)

def to_homo(points, euclid_dim = 2):
    """Conver points to homogenous point.
    Args:
        points: Nx2 np.adarray points
        eudlid_dim: input points' euclid space. 2->3, 3->4
    
    Returns:
        homo_points: Nx3 points    
    """
    if euclid_dim != 2 and euclid_dim != 3:
        raise ValueError(f"Expect euclid_dim 2 or 3, got {euclid_dim}")
    if len(points.shape) < 2:
        points = points.reshape( (-1, 2) )

    if points.shape[1] == euclid_dim+1:
        return points
    return np.hstack( [points, np.ones( (points.shape[0], 1) )] )

def enforce_singularity(F):
    """Force rank2 constraint on F."""
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

def get_epi_error(p1, p2, F):
    """Calculate distandce between p1, p2 with its corresponding epipolar line. 
    I have looked at my previous 16-720 hw5 for calculating epipolar error.  

    Args: 
        p1: Nx3 array
        p2: Nx3 array
    
    Returns:
        error: Nx1 array
    """
    p1, p2 = to_homo(p1), to_homo(p2)
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

def get_reprojection_error(P1, P2, pts1, pts2, pts_3d):
    """Calculate L2 reprojection error."""
    frame1_error = np.linalg.norm( normalize_points(pts1)-normalize_points(to_homo(pts_3d, 3)@P1.T), axis = 1 )
    frame2_error = np.linalg.norm( normalize_points(pts2)-normalize_points(to_homo(pts_3d, 3)@P2.T), axis = 1 )
    return (frame1_error**2).sum() + (frame2_error**2).sum()

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
    p1_hat, p2_hat = p1@T1.T, p2@T2.T

    A = np.matmul(p2_hat[..., None], p1_hat[:, None, :]) 
    A = A.reshape((-1,9))

    _, _, VT = np.linalg.svd(A)
    F_hat = VT[ -1 ].reshape((3, 3))
    F_hat = enforce_singularity(F_hat)

    # Un-normalize
    F = (T2.T).dot(F_hat).dot(T1)
    F = F/F[2,2]
    
    return F

def triangulate(P1, P2, pts1, pts2):
    """Triangulation to get 3D coordinates.

    Args:
        p1: 3x4 camera parameter
        p2: 3x4 camera parameter
        pts1: Nx3 points
        pts2: Nx3 points

    Returns:
        _3d_points: Nx4 points in 3D    
    """
    pts1_skew = to_skew(pts1)
    pts2_skew = to_skew(pts2)
    cons1 = pts1_skew@P1
    cons2 = pts2_skew@P2
    
    _3d_points = np.empty( (0, 3) )
    for c1, c2 in zip(cons1, cons2):
        A = np.vstack( [c1[:2], c2[:2]] )
        _, _, VT = np.linalg.svd(A)
        p3d = VT[-1]
        p3d = (p3d/p3d[-1])[:3]
        _3d_points = np.vstack([_3d_points, p3d] )

    return _3d_points