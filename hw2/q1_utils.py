import numpy as np
import cv2 as cv

_RED = [0, 0, 255]
_GREEN = [0, 255, 0]
_BLUE = [255, 0, 0]

def read_2d_3d_corres(correspondence_path):
    """Read correspoindendce in string and return np array.

    Args:
        correspondence_path: path to .txt file for correspondences.
    Returns:
        np.array(correspondences): np.array for correspondences.
    """

    correspondences = []
    with open(correspondence_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            correspondences.append( [ float(num) for num in line] )
    return np.array(correspondences)

def get_constraints( points_2d, points_3d ):
    """Convert set of 2d 3d point pairs into DLT contstraints.

    Args:
        points_2d: 2d points, shape: Nx2.
        points_3d: 3d points, shape: Nx3.

    Returns:
        _A: constaints for DLT in form of Ax = 0.
    """
    _x, _y = points_2d[:, 0, None], points_2d[:, 1, None]
    _X, _Y, _Z = points_3d[:, 0, None], points_3d[:, 1, None], points_3d[:, 2, None]
    _ones = np.ones( (_x.shape[0], 1) )
    _zeros = np.zeros( (_x.shape[0], 1) )

    cons1 = np.hstack( [ _X, _Y, _Z, _ones,  _zeros, _zeros, _zeros, _zeros, -_x*_X, -_x*_Y, -_x*_Z, -_x] )
    cons2 = np.hstack( [ _zeros,  _zeros, _zeros, _zeros, _X, _Y, _Z, _ones, -_y*_X, -_y*_Y, -_y*_Z, -_y] )
    _A = np.vstack( [cons1, cons2] )
    return _A

def compute_P( correspondences ):
    """Calculate 3d to 2d projection matrix with SVD using 2d 3d point pairs.

    Args:
        Correspoindences: 2d and 3d point pairs. shape = Nx5

    Returns:
        _P: projection matrix.
    """

    points_2d = correspondences[:, :2]
    points_3d = correspondences[:, 2:]
    _A = get_constraints( points_2d, points_3d )
    _, _, VT = np.linalg.svd( _A )
    _P = np.reshape(VT[-1]/VT[-1][-1], (3, 4))

    # Check reprojection error in pixel
    # for _2d, _3d in zip( points_2d, points_3d ):
    #     proj_2d = np.dot( P, np.array( [_3d[0], _3d[1], _3d[2], 1] ) )
    #     print( f"Reprojection error: {abs(_2d - (proj_2d/proj_2d[-1])[:-1])} pixels."  )
    
    return _P

def projection(_3d_points, _P):
    """_2d_points = _3d_points.dot(_P.T).

    Args: 
        _3d_points: nx3.
        _P: projection metrics, 3x4.
    
    Return:
        _2d_points: nx2 normalized points.
    """
    _3d_points = np.array(_3d_points)
    _3d_points = np.hstack( [_3d_points, np.ones( (_3d_points.shape[0], 1) )] )
    _2d_points = np.dot( _3d_points, _P.T )

    return np.divide(_2d_points, _2d_points[:, -1, None])[:, :-1]

def draw_points(image, points):
    image = image.copy()
    points = points.astype(int)
    for p in points:        
        image = cv.circle(image, p, radius=2, color=_RED, thickness=-1)
    return image

def draw_lines( image, start, end, thickness = 10 ):
    image = image.copy()
    start = start.astype(int)
    end = end.astype(int)

    for s, e in zip(start, end):
        image = cv.line(image, s, e, _BLUE, thickness)
    return image