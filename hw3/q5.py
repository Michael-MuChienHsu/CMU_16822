import numpy as np
import argparse
import cv2 as cv
from q1_utils import to_homo, show_epipolar_line
from q2_utils import get_F_ransac

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--point", type = int, default=8)
    parser.add_argument("-o", "--object", type = str, default="cup")
    args = parser.parse_args()
    object_name = args.object
    point_num = args.point

    im1 = cv.imread(f"./data/q5/{object_name}1.jpg")
    im1 = cv.resize(im1, (900, 1200))

    im2 = cv.imread(f"./data/q5/{object_name}2.jpg")
    im2 = cv.resize(im2, (900, 1200))
    
    sift = cv.SIFT_create()

    gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)    
    gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)    

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    p1 = to_homo( np.array([ kp1[match.queryIdx].pt for match in matches ]) )
    p2 = to_homo( np.array([ kp2[match.trainIdx].pt for match in matches ]) )

    F, best_fit_p1, inliner_record = get_F_ransac( p1, p2, point_num=point_num )
    im1, im2 = show_epipolar_line(im1, im2, F, best_fit_p1)
    cv.imwrite(f"q5_{object_name}1_{point_num}.png", im1)
    cv.imwrite(f"q5_{object_name}2_{point_num}.png", im2)