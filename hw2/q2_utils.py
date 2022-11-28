import numpy as np
import cv2 as cv
from scipy.linalg import cholesky

_RED = [0, 0, 255]
_GREEN = [0, 255, 0]
_BLUE = [255, 0, 0]
_COLORS = [ _BLUE, _GREEN, _RED, (0, 255, 255), (255, 255, 0)]

def get_color(i):
    """Get color code from i."""
    if i >= len(_COLORS):
        i = -1
    return _COLORS[i]

def draw_q2a_notaion(image, _points):
    """Draws sets of parallel lines for q2a."""

    assert len(_points) == 12
    image = image.copy()
    start_points = [ _points[2*i] for i in range(len(_points)//2) ]
    end_points = [ _points[2*i+1] for i in range(len(_points)//2) ]
    
    for i, (s, e) in enumerate( zip(start_points, end_points) ):
        cv.line(image, s, e, get_color(i//2), thickness=5)

    cv.imwrite( "q2a_labeled_image.png", image )

def draw_q2a_vanishing_points(vanishing_points, image, write_path = "q2a_vanishing_points.png"):
    """Draws vanishing points and connects vanishing points."""

    x_pos = [ int(p[0]) for p in vanishing_points ]
    y_pos = [ int(p[1]) for p in vanishing_points ]

    x_offset = abs(min([0]+x_pos)) + 100
    y_offset = abs(min([0]+y_pos)) + 100

    # udpate x_pos and y_pos with offsets
    x_pos = [pos+x_offset for pos in x_pos]
    y_pos = [pos+y_offset for pos in y_pos]
    
    canvas_x = max( image.shape[0], max(x_pos) ) + 100
    canvas_y = max( image.shape[1], max(y_pos) ) + 100

    H = np.array([ [1, 0, x_offset], 
                   [0, 1, y_offset], 
                   [0, 0, 1] ]).astype(np.float32)
    canvas = cv.warpPerspective( image, H, (canvas_x, canvas_y) )
    canvas[ canvas == 0 ] = 255

    for i in range(len(x_pos)):
        sx, sy = x_pos[(i+1)%3], y_pos[(i+1)%3]
        ex, ey = x_pos[(i+2)%3], y_pos[(i+2)%3]
        cv.line(canvas, (sx, sy), (ex, ey), get_color(i), thickness=5)
    for i in range(len(x_pos)):
        cv.circle( canvas, (x_pos[i], y_pos[i]), 20, get_color(i), thickness=-1 )

    cv.imwrite( write_path, canvas)

def points_to_lines(points):
    """Every 2 points in image defines a line in projectiove space. This 
    function converts N points in to N//2 homoguous lines.

    Args:
        points: Nx2 points, list
    Return:
        lines: (N//2)x2 lines, ndarray, normalized
    """
    if len(points) % 2 != 0:
        raise ValueError(f"Expect even len(points), got {len(points)}")

    lines = np.empty((0, 3))
    for i in range(len(points)//2):
        p1 = np.array(points[2*i] + [1])
        p2 = np.array(points[2*i+1] + [1])
        _line = np.cross(p1, p2)

        lines = np.vstack( [lines, _line/_line[-1]] )
    
    return lines

def get_intersection(l1, l2):
    """Given 2 lines l1 and l2, calculate intersection between l1 and l2.

    Args:
        l1: line, ndarray
        l2: line, ndarray
    
    Return:
        intersect: normalized point.
    """
    intersect = np.cross(l1, l2)
    return intersect/intersect[-1]

def get_orthogonal_constraint(p1, p2):
    """Get constraints from 2 vanish points to solve IAC.

    Args:
        p1: normalized vanishing point.
        p2: normalized vanishing point.

    Returns:
        constraint: ndarray. Shape = 1x4.
    """
    p1_x, p1_y, _ = p1
    p2_x, p2_y, _ = p2

    return np.array([p1_x*p2_x + p1_y*p2_y, 
                     p1_x + p2_x,
                     p1_y + p2_y, 
                     1]).reshape(1, 4)

def get_K(vanishing_points):
    """Solve Ax = 0 with DLT.

    Args:
        vanishing_points: 3 vanishing points.

    Returns:
        Normalized intrinsic Matrix K.

    """
    assert len(vanishing_points) == 3
    constraints = np.empty((0, 4))
    for i in range(3):
        cons = get_orthogonal_constraint( vanishing_points[i], vanishing_points[(i+1)%3] )
        constraints = np.vstack( [constraints, cons] )

    _, _, VT = np.linalg.svd( constraints )
    w1, w2, w3, w4 = VT[-1]
    IAC = np.array( [[w1, 0, w2], 
                     [0, w1, w3],
                     [w2, w3, w4]] )
    IAC = IAC/IAC[2][2]
    K = np.linalg.inv( cholesky(IAC) )

    # Sanity check
    for i in range(3):
        assert abs(vanishing_points[i].T.dot(IAC).dot(vanishing_points[(i+1)%3])) < 1e-5

    return K/K[2][2]

def draw_poly( image, points, color ):
    """Draw a polygon given points.
    Args: 
        image: the image to draw on.
        points: sequential points in list.
    
    Returns:
        image: the image with filled polygon.
    """
    image = image.copy()
    # cv.fillPoly(image, pts = [np.array(points)], color = color)
    cv.polylines(image, pts = [np.array(points)], isClosed = True, color = color, thickness=4)
    for i, p in enumerate(points):
        image = cv.putText(image, str(i), tuple(p), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

def get_homography_constraint(p1, p2):
    """DLT on p1, p2 to make constraints for solving homography to map p2 to p1."""

    x_prime, y_prime = p1[0], p1[1]
    x_, y_= p2[0], p2[1]

    cons = [ [ -x_, -y_, -1, 0, 0, 0, x_*x_prime, y_*x_prime, x_prime ],
                  [ 0, 0, 0, -x_, -y_, -1, x_*y_prime, y_*y_prime, y_prime ] ]

    return np.array(cons)

def normalize_homo(point):
    """Normalize point or line to depth = 1."""
    return point/point[-1]

def get_H(plane_points):
    """Get homography between standard square and plane points.

    Args:
        plane_points: 4 euclid points that of a square in image.

    Return:
        _H: Normalized homography for the the plane.
    """
    standard_square_points = np.array([ [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1] ]) # clock wise

    A = np.empty( (0, 9) )

    x_prime, y_prime = plane_points.T
    x_, y_, _= standard_square_points.T
    x_prime, y_prime, x_, y_ = x_prime.reshape((4, 1)), y_prime.reshape((4, 1)), x_.reshape((4, 1)), y_.reshape((4, 1)) 
    ones = np.ones_like(x_prime)
    zeros = np.zeros_like(x_prime)
    A = np.vstack([ 
            np.hstack([ -x_, -y_, -ones, zeros, zeros, zeros, x_*x_prime, y_*x_prime, x_prime ]),
            np.hstack([ zeros, zeros, zeros, -x_, -y_, -ones, x_*y_prime, y_*y_prime, y_prime ])
         ])
    _, _, VT = np.linalg.svd(A)

    H2to1 = VT[-1].reshape((3, 3))

    # Unit test, HS should be same as plane point.
    # HX = np.array(standard_square_points).dot(H2to1.T)
    # HX = HX[:, :-1]/HX[:, -1: None]
    # print( plane_points )
    # print( HX )
    # print(f"error: {np.linalg.norm( HX-np.array(plane_points))}")
    # assert np.linalg.norm( HX-np.array(plane_points)) < 1e-5
    H2to1 = H2to1/H2to1[2][2]
    return H2to1

def get_IAC_constraint(H):
    """DLT on H to estimate IAC.
    2 constraints are: (1) h1.T w h2.T = 0 and (2)  h1.T w h1.T - h2.T w h2.T = 0.
    Where h1, h2, h3 are columns of H.
    """

    h1, h2, _ = H.T
    h1 = h1.reshape((3, 1))
    h2 = h2.reshape((3, 1))
    cons1 = h1.dot(h2.T).reshape((1, 9))
    cons2 = h1.dot(h1.T).reshape((1, 9)) - h2.dot(h2.T).reshape((1, 9))
    return np.vstack( [cons1, cons2] )

def get_plane_noraml( square_points, K ):
    """Given 4 homonguous points of a square, calculate the vanishing line.
    p1, p2, p3, p4 are in clockwise order.

    Args:
        square_points: 4 homoguous points in clockwise order.
        p1---p2
        |     |
        p4---p3

    Returns:
        Normalized plane normal vector (orientation).
    """
    p1, p2, p3, p4 = np.array( square_points )
    l1 = np.cross( p1, p2 )
    l2 = np.cross( p4, p3 )
    vp1 = normalize_homo(np.cross( l1, l2 ))
    d1 = np.linalg.inv(K).dot(vp1)

    l1 = np.cross( p1, p4 )
    l2 = np.cross( p2, p3 )
    vp2 = normalize_homo(np.cross( l1, l2 ))
    d2 = np.linalg.inv(K).dot(vp2)

    plane_normal = np.cross(d1, d2 )
    # print(vp1, vp2)
    return plane_normal/plane_normal[-1]

def cosine(u, v):
    return u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v))
    
def get_plane_angle(p1, p2, K):
    """Calulate angles between 2 plane given intrinsic K.

    Args:
        p1: 4 homonguous points in image that builds a plane. Clockwise.
        p2: 4 homonguous points in image that builds a plane. Clockwise.
    """
    p1_normal = get_plane_noraml( p1, K )
    p2_normal = get_plane_noraml( p2, K )

    cos_theta = cosine( p1_normal, p2_normal )

    radius = np.arccos(cos_theta)
    return radius/np.pi*180