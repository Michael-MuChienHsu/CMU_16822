import cv2
import numpy as np
from utils import *
import os
import argparse

_RED = [0, 0, 255]
_GREEN = [0, 255, 0]
_BLUE = [255, 0, 0]

def click_event(event, x, y, flags, params):
    """ Directly draw line on global param: label_img"""
    if event == cv2.EVENT_LBUTTONDOWN: 
        color = _GREEN

        # homoguos points
        points.append( [x, y, 1] ) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(label_img, str(len(points)), (x, y), font, 1, color, 3)
        cv2.circle(label_img, (x,y), radius=3, color=_GREEN, thickness =-1)

        cv2.imshow('image', label_img)

def add_mouse_response(img_name, reset = False):
    """Catch mouse response event on img_name"""
    global points
    if reset:
        points = []

    # get point pairs
    cv2.setMouseCallback(img_name, click_event) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize_line( line ):
    return line/line[-1]

def get_H(p1, p2):
    """Get homography by solving DLT.
    
    Args:
        p1: 4 points in euclid space on image 1. Shape 4x2.
        p2: 4 points in euclid space on image 2. Shape 4x2.

    Returns:
        H: Homography to warp image 2 to image 1.
    """

    A = np.empty( (0, 9) )

    x_prime, y_prime, _ = p1.T
    x_, y_, _ = p2.T
    x_prime, y_prime, x_, y_ = x_prime.reshape((4, 1)), y_prime.reshape((4, 1)), x_.reshape((4, 1)), y_.reshape((4, 1)) 
    ones = np.ones_like(x_prime)
    zeros = np.zeros_like(x_prime)
    A = np.vstack([ 
            np.hstack([ -x_, -y_, -ones, zeros, zeros, zeros, x_*x_prime, y_*x_prime, x_prime ]),
            np.hstack([ zeros, zeros, zeros, -x_, -y_, -ones, x_*y_prime, y_*y_prime, y_prime ])
         ])
    _, _, VT = np.linalg.svd(A)

    H2to1 = VT[-1].reshape((3, 3))

    return H2to1

def composite_img(H, background, target):
    composite_img = cv2.warpPerspective( target, H, background.shape[:2][::-1] )
    composite_img[ composite_img == 0 ] = background[ composite_img == 0 ]

    return composite_img.astype(np.uint8)

if __name__=="__main__":
    # example: python .\q3.py desk-normal.png desk-perspective.png
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('target', type=str, help='img_name with .jpg')
    parser.add_argument('background', type=str, help='target with .jpg')

    args = parser.parse_args()
    
    target = args.target
    background = args.background
    
    target_path = os.path.join("./data/q3/", target)
    background_path = os.path.join("./data/q3/", background)
    assert os.path.isfile( target_path )
    assert os.path.isfile( background_path )

    points = []
    img = cv2.imread(background_path)
    target_img = cv2.imread(target_path)
    target_y, target_x= target_img.shape[0], target_img.shape[1]
    cv2.imshow('image', img)

    # Q3: affine transform.
    label_img = img.copy()
    print( "Click 4 points in clock-wise order." )
    add_mouse_response("image")
    cv2.imwrite(f"./output_fig/q3_{background}_anotate.png", label_img)
    
    target_points = np.array( [ [0, 0, 1], [target_x, 0, 1], [target_x, target_y, 1], [0, target_y, 1]  ] )
    desk_points = np.array(points[:4])

    H = get_H( desk_points, target_points )

    combine_img = composite_img( H, img, target_img )
    cv2.imshow("comb", combine_img)
    cv2.imwrite(f"./output_fig/q3_combined_{background}", combine_img)
    cv2.waitKey(0)