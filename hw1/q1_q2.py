import cv2
import numpy as np
from utils import *
import argparse

_RED = [0, 0, 255]
_GREEN = [0, 255, 0]
_BLUE = [255, 0, 0]

def click_event(event, x, y, flags, params):
    """ Directly draw line on global param: label_img"""
    if event == cv2.EVENT_LBUTTONDOWN: 
        if len(points)<5:
            color = _BLUE
        else:
            color = _RED

        # homoguos points
        points.append( [x, y, 1] ) 
        if len(points)%2 == 0 and len(points)>0:
            cv2.line(label_img, points[-2][:2], points[-1][:2], color, 2)

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
    """Normalize a homogeneous line to image plane."""
    return line/line[-1]

def end_point_pair_to_line_normal(point_pair):
    """Cross product on 2 homogeneous points to get a homegeneous line."""
    p1, p2 = point_pair
    l = np.cross(np.array(p1), np.array(p2))

    assert l.dot( p1 ) == 0
    assert l.dot( p2 ) == 0
    return normalize_line(l)

def get_affine_rectification_H(l1, l2, l3, l4):
    """Calculate H_affine by estimating 2 pairs of parallel lines.
    
    Args: 
        l1: a homogeous line that paralles to l2.
        l2: a homogeous line that paralles to l1.
        l3: a homogeous line that paralles to l4.
        l4: a homogeous line that paralles to l3.

    Returns:
        H: Affine homography
    """

    inf_p1 = normalize_line(np.cross( l1, l2 ))
    inf_p2 = normalize_line(np.cross( l3, l4 ))
    # print(inf_p1, inf_p2)
    inf_line = normalize_line(np.cross(inf_p1, inf_p2))

    H = np.eye(3)
    H[2] = inf_line

    return H

def affine_rectification(img, end_point_pairs):
    """Affine transform an image with 4 point pairs that define 2 set of parallel lines.
    
    Args:
        img: openv image
        end_point_pairs: 4 pair of end homogenous points. shape: 4x2x3
    
    Retusns:
        H_A: Affine Homography
        warped_img: Affine transformed image.
    """
    line_1, line_2, line_3, line_4 = end_point_pairs

    line_1 = end_point_pair_to_line_normal(line_1)
    line_2 = end_point_pair_to_line_normal(line_2)
    line_3 = end_point_pair_to_line_normal(line_3)
    line_4 = end_point_pair_to_line_normal(line_4)

    H_A = get_affine_rectification_H( line_1, line_2, line_3, line_4 )
    print( f"Before: {cosine(line_1, line_2)}, {cosine(line_3, line_4)}" )
    H_inv = np.linalg.inv(H_A)
    H_inv = H_inv/H_inv[2, 2]
    print(f"After: {cosine(H_inv.T.dot(line_1), H_inv.T.dot(line_2))}, {cosine(H_inv.T.dot(line_3), H_inv.T.dot(line_4))}")

    warped_img = MyWarp(img, H_A)
    return H_A, warped_img

def get_line_end_points(number_of_lines = 2):
    assert len(points)%2 == 0

    line_end_points = [ [points[2*i], points[2*i+1]] for i in range(len(points)//2) ][:2*number_of_lines]
    # print(f"point pairs: {line_end_points}")
    assert len(line_end_points) == 2*number_of_lines

    return line_end_points

def get_row_constraint( line_1, line_2 ):
    """Get row constraints for metric rectfication."""
    x1, y1, _ = normalize(line_1)
    x2, y2, _ = normalize(line_2)
    return np.array( [x1*x2, x1*y2+x2*y1, y1*y2] )

def metric_rectification( affine_warped_img, H_A, line_end_points ):
    """Metric rectification with 2 set of perpentdicular lines.
    
    Args:
        affine_warped_img: Affine rectified image.
        H_A: Affine homography.
        line_end_points: 2 pairs of perpendicular homogenous lines. Shape: 4x2x3
    
    Returns:
        H_S: Similarity Homography.
        img: Image restored upto similarity transform.
    """
    line_1, line_2, line_3, line_4 = line_end_points
    line_1 = end_point_pair_to_line_normal(line_1)
    line_2 = end_point_pair_to_line_normal(line_2)
    line_3 = end_point_pair_to_line_normal(line_3)
    line_4 = end_point_pair_to_line_normal(line_4)
    print( f"Before: {cosine(line_1, line_2)}, {cosine(line_3, line_4)}" )

    # warp lines to affine
    H_inv = np.linalg.inv(H_A)
    H_inv = H_inv/H_inv[2, 2]
    line_1 = H_inv.T.dot(line_1)
    line_2 = H_inv.T.dot(line_2)
    line_3 = H_inv.T.dot(line_3)
    line_4 = H_inv.T.dot(line_4)

    row_constraint_1 = get_row_constraint( line_1, line_2 )
    row_constraint_2 = get_row_constraint( line_3, line_4 )
    coef = np.cross( row_constraint_1, row_constraint_2 )
    a = coef[0]/coef[2]
    b = coef[1]/coef[2]

    C_inf_prime = np.array( [[ a, b, 0 ], 
                             [ b, 1, 0 ],
                             [ 0, 0, 0 ]] )

    assert line_1.dot(C_inf_prime).dot(line_2) < 1e-5 
    assert line_3.dot(C_inf_prime).dot(line_4) < 1e-5 

    U, S, VT = np.linalg.svd(C_inf_prime)    

    S[-1] = 1
    S = np.sqrt(1/S)
    H_S = np.diag( S ).dot(VT)

    H_inv = np.linalg.inv(H_S)
    H_inv = H_inv/H_inv[2, 2]
    print(f"After: {cosine(H_inv.T.dot(line_1), H_inv.T.dot(line_2))}, {cosine(H_inv.T.dot(line_3), H_inv.T.dot(line_4))}")
    H = np.matmul(H_A, H_S)
    # if np.linalg.det( H[:2][:2] ) == -1:
    #     H[0][0], H[0][1] = H[0][1], H[0][0]
    #     H[1][0], H[1][1] = H[1][1], H[1][0]

    # return H_S, MyWarp(affine_warped_img, np.matmul(H_A, H_S))
    return H_S, MyWarp(affine_warped_img, np.matmul(H_S, H_A))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('img_name', type=str, help='img_name  with .jpg that in ./data/q1')
    args = parser.parse_args()
    
    img_name = args.img_name
    points = []
    img = cv2.imread(f"./data/q1/{img_name}", 1)
    # img = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2))  
    cv2.imshow('image', img)

    # Q1: affine transform.
    label_img = img.copy()
    print("Click on the image to annotate 4 lines that l1 // l2 and l3 // l4.")
    add_mouse_response("image")
    line_end_points = get_line_end_points()
    cv2.imwrite(f"./output_fig/q1_{img_name}_anotate.png", label_img)

    # Predefined points for debug
    # line_end_points = [[[93, 160, 1], [582, 159, 1]], [[131, 359, 1], [546, 360, 1]], [[582, 161, 1], [551, 358, 1]], [[94, 158, 1], [125, 361, 1]]]
    H_A, affine_warped_img = affine_rectification(img, line_end_points)
    # print(H_A)
    cv2.imwrite(f"./output_fig/q1_{img_name}_affine_rect.png", affine_warped_img)
    cv2.imshow('q1_result', affine_warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Q1 done.")



    # Q2: Similarity transform based on Q1 result.
    cv2.imshow('image', img)
    label_img = img.copy()
    print("Click on the image to annotate 4 lines that l1 T l2 and l3 T l4.")
    add_mouse_response("image", reset=True)
    cv2.imwrite(f"./output_fig/q2_{img_name}_annotate.png", label_img)

    line_end_points = get_line_end_points()
    lebel_affine_img = MyWarp(label_img, H_A)
    cv2.imwrite(f'./output_fig/q2_{img_name}_annotate_affine_rect.png', lebel_affine_img)
    cv2.waitKey(0)

    # Predefined points for debug
    # line_end_points = [[[130, 360, 1], [550, 357, 1]], [[339, 359, 1], [338, 208, 1]], [[308, 127, 1], [368, 187, 1]], [[309, 183, 1], [366, 132, 1]]]
    H_S, similarity_warped_img = metric_rectification(img, H_A, line_end_points)
    cv2.imshow('q2_result',similarity_warped_img)
    cv2.waitKey(0)
    cv2.imwrite( f'./output_fig/q2_{img_name}_similarity_rect.png', similarity_warped_img )
    print("Q2 done")