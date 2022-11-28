"""Your goal in this question is to implement initialization for incremental SfM."""
from q1_utils import *
from utils import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def q1_main(im1, im2, K, cor):
    """Baseline Reconstruction.
    1. Get F through RANSAC. E = K2.T F K1.
    2. Recover R t from E: E = t.cross(R). R&A P.258
    3. Triangulation using P1 = K1[R1|t1], P2 = K2[R2|t2]
    4. Retuen 3D reconstruction

    Args:
        im1: cv2 image
        im2: cv2 image
        K: intrinsic matrix dictionary
        cor: noisy correspondences dictionary
 
    Returns:
        P1: Estimated camera parameter 1.
        P2: Estiamted camera parameter 2.
        estimated_3d_points: Triangulated 3D points using P1 and P2.
    """

    K1, K2, pts1, pts2 = K.item()["K1"], K.item()["K2"], cor["pts1"], cor["pts2"]
    F, = get_F_ransac(pts1, pts2, num_iter=10000)
    E = K2.T@F@K1
    R, t, P1, P2, estimated_3d_points = get_P1_P2( to_homo(pts1), to_homo(pts2), K1, K2, E )
    
    return R, t, P1, P2, estimated_3d_points

if __name__ == "__main__":
    im1 = cv.imread("./data/monument/im1.jpg")
    im2 = cv.imread("./data/monument/im2.jpg")
    K = np.load("./data/monument/intrinsics.npy", allow_pickle=True)
    cor = np.load("./data/monument/some_corresp_noisy.npz", allow_pickle=True)

    R, t, P1, P2, estimated_3d_points = q1_main(im1, im2, K, cor)
    print(f"R: \n{R}")
    print(f"t: {t}")

    assert np.linalg.det(R) > 0

    # plot
    X, Y, Z = estimated_3d_points.T
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z )
    plt.show()