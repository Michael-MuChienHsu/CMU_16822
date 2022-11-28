import numpy as np
from q1_utils import *
import cv2 as cv
import argparse

def q1a():
    # x = np.array([970 , 779, 1])
    # working on chair 
    chair_im1 = cv.imread("./data/q1a/chair/image_1.jpg")
    chair_im2 = cv.imread("./data/q1a/chair/image_2.jpg")
    chair_cor = np.load("./data/q1a/chair/chair_corresp_raw.npz")    
    chair_k = np.load("./data/q1a/chair/intrinsic_matrices_chair.npz")
    p1, p2 = chair_cor["pts1"], chair_cor["pts2"]
    K1, K2 = chair_k["K1"], chair_k["K2"]

    F = eightpoint_F(p1, p2)
    E = get_E_from_F( K1, K2, F )
    E = E/E[-1][-1]

    im1_points = to_homo(np.array([ p1[i] for i in np.random.choice(p1.shape[0], 5) ]+[[970 , 779]]))
    im2_points = to_homo(np.array([ p2[i] for i in np.random.choice(p2.shape[0], 5)]))
    # for _p1, _p2 in zip(im1_points[:-1], im2_points):
        # print( _p2.T.dot(F).dot(_p1) )
    
    im1, im2 = show_epipolar_line( chair_im1, chair_im2, F, im1_points )
    cv.imwrite( "q1a_chair1.png", im1 )
    cv.imwrite( "q1a_chair2.png", im2 )

    print("Foundamental Matrix for chair:")
    print(F)
    print("Essential Matrix for chair:")
    print(E)


    # working on teddy 
    # x = np.array([790, 467, 1])
    teddy_im1 = cv.imread("./data/q1a/teddy/image_1.jpg")
    teddy_im2 = cv.imread("./data/q1a/teddy/image_2.jpg")
    teddy_cor = np.load("./data/q1a/teddy/teddy_corresp_raw.npz")    
    teddy_k = np.load("./data/q1a/teddy/intrinsic_matrices_teddy.npz")
    p1, p2 = teddy_cor["pts1"], teddy_cor["pts2"]
    K1, K2 = teddy_k["K1"], teddy_k["K2"]
    F = eightpoint_F(p1, p2)
    E = get_E_from_F( K1, K2, F )
    
    im1_points = to_homo(np.array([ p1[i] for i in np.random.choice(p1.shape[0], 5) ]+[[790, 467]]))
    im1, im2 = show_epipolar_line( teddy_im1, teddy_im2, F, im1_points)
    cv.imwrite( "q1a_teddy1.png", im1 )
    cv.imwrite( "q1a_teddy2.png", im2 )
    print("Foundamental Matrix for teddy:")
    print(F)
    print("Essential Matrix for teddy:")
    print(E)


def q1b():
    # working on toybus
    im1 = cv.imread("./data/q1b/toybus/image_1.jpg")
    im2 = cv.imread("./data/q1b/toybus/image_2.jpg")
    correspond = np.load("./data/q1b/toybus/toybus_7_point_corresp.npz")
    p1, p2 = correspond["pts1"],  correspond["pts2"]    
    F = sevenpoint_F(p1, p2)

    # Pick best F
    p1, p2 = to_homo(p1), to_homo(p2)
    best_F = pick_sevenpoint_F(F, p1, p2)

    im1_points = to_homo(np.array([ p1[i] for i in np.random.choice(p1.shape[0], 5, replace=False) ]))
    im1, im2 = show_epipolar_line( im1, im2, best_F, im1_points )
    cv.imwrite( "q1b_toybus1.png", im1 )
    cv.imwrite( "q1b_toybus2.png", im2 )
    print("best toy bus F:")
    print(best_F)

    # working on toytrain
    im1 = cv.imread("./data/q1b/toytrain/image_1.jpg")
    im2 = cv.imread("./data/q1b/toytrain/image_2.jpg")
    correspond = np.load("./data/q1b/toytrain/toytrain_7_point_corresp.npz")
    p1, p2 = correspond["pts1"],  correspond["pts2"]
    
    F = sevenpoint_F(p1, p2)
    best_F = pick_sevenpoint_F(F, p1, p2)
    im1_points = to_homo(np.array([ p1[i] for i in np.random.choice(p1.shape[0], 5, replace=False) ]))
    p1, p2 = to_homo(p1), to_homo(p2)
    im1, im2 = show_epipolar_line( im1, im2, best_F, im1_points )
    cv.imwrite( "q1b_toytrain1.png", im1 )
    cv.imwrite( "q1b_toytrain2.png", im2 )
    print("best toy train F:")
    print(best_F)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--a", type = bool)
    parser.add_argument("-b", "--b", type = bool)
    args = parser.parse_args()

    if args.a:
        q1a()
    if args.b:
        q1b()