import numpy as np
import cv2 as cv
import matplotlib.pylab as plt
from q1_utils import *
from q2_utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--point", type = int)
    parser.add_argument("-o", "--object", type = str, default="toybus")
    args = parser.parse_args()
    object_name = args.object
    point_num = args.point

    matched_points = np.load(f"./data/q1b/{object_name}/{object_name}_corresp_raw.npz")
    pts1 = to_homo(matched_points["pts1"])
    pts2 = to_homo(matched_points["pts2"]) 

    F, best_fit_p1, inliner_record = get_F_ransac( pts1, pts2, point_num=point_num )

    print(F)
    # plot lnlier vs iteration
    plt.plot([i+1 for i in range(len(inliner_record))], inliner_record, linewidth=2, markersize=1)
    plt.savefig(f"q2_{object_name}_inliner_{point_num}point_algo.png")

    # plot epipolar line
    im1 = cv.imread(f"./data/q1b/{object_name}/image_1.jpg")
    im2 = cv.imread(f"./data/q1b/{object_name}/image_2.jpg")
    im1, im2 = show_epipolar_line(im1, im2, F, best_fit_p1)
    cv.imwrite(f"q2_{object_name}1_{point_num}.png", im1)
    cv.imwrite(f"q2_{object_name}2_{point_num}.png", im2)


