import numpy as np
import cv2 as cv
from q1_utils import to_homo
import matplotlib.pyplot as plt

def to_skew(pts):
    x, y, z = pts.T
    x, y, z = x[:, None], y[:, None], z[:, None]
    zeros = np.zeros_like( x )
    skew = np.hstack( [zeros, -z, y,
                       z, zeros, -x,
                       -y, x, zeros ] )
    return skew.reshape(-1, 3, 3)

def triangulate(p1, p2, pts1, pts2):
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
    cons1 = pts1_skew@p1
    cons2 = pts2_skew@p2
    
    _3d_points = np.empty( (0, 3) )
    for c1, c2 in zip(cons1, cons2):
        A = np.vstack( [c1[:2], c2[:2]] )
        _, _, VT = np.linalg.svd(A)
        p3d = VT[-1]
        p3d = (p3d/p3d[-1])[:3]
        _3d_points = np.vstack([_3d_points, p3d] )

    return _3d_points

if __name__ == "__main__":

    p1 = np.load("./data/q3/P1.npy")
    p2 = np.load("./data/q3/P2.npy")
    pts1 = to_homo(np.load("./data/q3/pts1.npy")) 
    pts2 = to_homo(np.load("./data/q3/pts2.npy"))

    im1 = cv.imread("./data/q3/img1.jpg")
    im2 = cv.imread("./data/q3/img2.jpg")

    X, Y, Z = triangulate(p1, p2, pts1, pts2).T
    color = [ (im1[int(y), int(x)]/255)[::-1] for (x, y, _) in pts1]

    print(X, Y, Z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z , c = color)
    plt.show()