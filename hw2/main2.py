import numpy as np
import cv2 as cv
from q2_utils import *
from scipy.linalg import cholesky

def click_event(event, x, y, flags, params):
    """ Directly draw line on global param: label_img"""
    if event == cv.EVENT_LBUTTONDOWN: 
        points.append( [x, y] )
        cv.circle( label_img, (x, y), radius=2, color=get_color(1), thickness=-1 )
        if len(points) % 2 == 0:
            cv.line(label_img, points[-2], points[-1], get_color(2), thickness=5)

        cv.imshow('label_img', label_img)

def add_mouse_response(img_name, reset = False):
    """Catch mouse response event on img_name."""

    global points
    if reset:
        points = []

    cv.setMouseCallback(img_name, click_event) 
    cv.waitKey(0)
    cv.destroyAllWindows()

def q2_a():
    """Perform q2_a, use default set of points to reproduce. Also can set annotate to True to label points."""
    global points
    points = []
    global label_img
    
    image_path = "./data/q2a.png"
    image = cv.imread(image_path)
    cv.imwrite("q2a_origin_image.png", image)

    if annotate:
        # annotate your own points!
        label_img = image.copy()
        cv.imshow( "label_img", label_img )
        add_mouse_response("label_img")
        draw_q2a_notaion(image, points)
        print(points)
    else:
        # Use provided points to reproduce.
        points = np.load("./data/q2/q2a.npy").ravel()
        points = [ [points[2*i], points[2*i+1]] for i in range(len(points)//2) ]

    lines = points_to_lines(points)

    vanishing_points = []
    for i in range(lines.shape[0]//2):
        intersect = get_intersection( lines[2*i], lines[2*i+1] ) 
        vanishing_points.append( intersect )
    draw_q2a_vanishing_points( vanishing_points, image )
    K = get_K(vanishing_points)
    print("q2a K is:")
    print(K)
    return K

def q2_b():
    global points
    points = []
    global label_img

    image_path = "./data/q2b.png"
    image = cv.imread(image_path)
    if annotate:
        cv.imwrite("q2b_origin_image.png", image)
        label_img = image.copy()
        cv.imshow( "label_img", label_img )
        print("Annotate points in clockwise.")
        add_mouse_response("label_img")
        print(points)
    else:
        # Use provided points to calibrate.
        plane_contour = np.load("./data/q2/q2b.npy").astype(int)

    for i, plane in enumerate(plane_contour):
        polygon_image = draw_poly( image, plane, get_color(i))
        cv.imwrite(f"q2b_poly_{i}.png", polygon_image)

    plane_constraints = np.empty((0, 9))
    _H_list = [] # Sanity check
    for plane in plane_contour:
        _H = get_H(plane)
        _H_list.append(_H) # Sanity check
        cons = get_IAC_constraint(_H)
        plane_constraints = np.vstack( [plane_constraints, cons] )
    _, _, VT = np.linalg.svd( plane_constraints )
    IAC = VT[-1].reshape((3, 3))
    IAC = IAC/IAC[2][2]

    # Sanity check
    for _h in _H_list:
        _h1, _h2, _ = _h.T
        assert _h1.dot(IAC).dot(_h2) < 1e-7
        assert abs(_h1.dot(IAC).dot(_h1) - _h2.dot(IAC).dot(_h2)) < 1e-7
        
    K = np.linalg.inv( cholesky(IAC) )

    K = K/K[2][2]
    print("q2b K:")
    print(K)

    homo_plane_contour = [ [ [point[0], point[1], 1] for point in plane ] for plane in plane_contour]
    angle = get_plane_angle(homo_plane_contour[0], homo_plane_contour[1], K)
    print(f"plane 1 and plane 2: {angle}")

    angle = get_plane_angle(homo_plane_contour[0], homo_plane_contour[2], K)
    print(f"plane 1 and plane 3: {angle}")

    angle = get_plane_angle(homo_plane_contour[1], homo_plane_contour[2], K)
    print(f"plane 2 and plane 3: {angle}")

    return K

if __name__ == "__main__":
    annotate = False
    K_a = q2_a()
    K_b = q2_b()
