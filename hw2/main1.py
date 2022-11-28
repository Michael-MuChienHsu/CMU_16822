import numpy as np
import cv2 as cv
from q1_utils import *
from q1_utils import _RED, _GREEN, _BLUE

def click_event(event, x, y, flags, params):
    """ Directly draw line on global param: label_img"""
    if event == cv.EVENT_LBUTTONDOWN: 
        points.append( [x, y] ) 
        cv.circle(label_img, (x, y), radius=2, color=_RED, thickness=-1)
        cv.putText(label_img, str(len(points)-1), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, _RED, 3)
        cv.imshow('cube', label_img)
        cv.imwrite("label_img.png", label_img)

def add_mouse_response(img_name, reset = False):
    """Catch mouse response event on img_name"""
    global points
    if reset:
        points = []

    cv.setMouseCallback(img_name, click_event) 
    cv.waitKey(0)
    cv.destroyAllWindows()


def q1_a():
    """Calculate P and draw bounding box and surface points on image."""
    correspondences_path = "./data/q1/bunny.txt"
    surface_path = "./data/q1/bunny_pts.npy"
    bbox_path = "./data/q1/bunny_bd.npy"

    # Read 2d 3d correspondence and compute P
    bunny_img = cv.imread("./data/q1/bunny.jpeg")
    correspondences = read_2d_3d_corres(correspondences_path)
    P = compute_P( correspondences )

    # Read bounding box in 3D, project with P and draw bbox 
    bbox_lines = np.load( bbox_path )
    start_points = bbox_lines[:, :3]
    end_points = bbox_lines[:, 3:]

    start_points = projection(start_points, P)
    end_points = projection(end_points, P)

    bbox_image = draw_lines( bunny_img, start_points, end_points )
    cv.imwrite("bbox_image.png", bbox_image)

    # Read 3d surface points, project with P and draw surface points
    surface_points = np.load(surface_path)
    surface_points_2d = projection( surface_points, P )
    surface_image = draw_points( bunny_img, surface_points_2d )
    cv.imwrite("surface_image.png", surface_image)

def q1_b():
    cube_path = "./data/q1/cube.jpg"
    cube_image = cv.imread(cube_path)
    global label_img
    global points
    label_img = cube_image.copy()

    _3d_point_position = np.array([[0, 0, 0], 
                                   [0, 0, 3],
                                   [3, 0, 0],
                                   [3, 0, 3],
                                   [3, 3, 0],
                                   [3, 3, 3]])
    if annotate:
        cv.imshow("cube", cube_image)
        add_mouse_response("cube")
        points = np.array(points)
    else:
        # Use these points to reprodice results. 
        points = [[ 58, 401], [ 44, 123], [328, 460], [331, 175], [437, 338], [445, 77]]

    correspondence = np.hstack( [points, _3d_point_position] )
    P = compute_P(correspondence)
    # Draw X on each plane on the cube:
    # start_points = [[0, 0, 0], [0, 0, 3], [3, 0, 0], [3, 0, 3], [0, 0, 3], [0, 3, 3]]
    # end_points =   [[3, 0, 3], [3, 0, 0], [3, 3, 3], [3, 3, 0], [3, 3, 3], [3, 0, 3]]

    # Draw CMU on cube
    start_points = [[0.5, 0, 0.5], [0.5, 0, 2.5], [1.5, 0, 0.5], [2.5, 0, 2.5],
                    [2.5, 2.5, 3], [0.5, 2.5, 3], [0.5, 0.5, 3], 
                    [3, 0.5, 2.5], [3, 0.5, 0.5], [3, 2.5, 0.5]]
    end_points =   [[0.5, 0,2.5], [1.5, 0, 0.5], [2.5, 0, 2.5], [2.5, 0, 0.5],
                    [0.5, 2.5, 3], [0.5, 0.5, 3], [2.5, 0.5, 3], 
                    [3, 0.5, 0.5], [3, 2.5, 0.5], [3, 2.5, 2.5]]

    start_points = projection(start_points, P)
    end_points = projection(end_points, P)

    cube_image_draw = draw_lines( cube_image, start_points, end_points, thickness = 3 )
    cv.imshow("cube_image_draw", cube_image_draw)
    cv.imwrite("cube_image_draw.png", cube_image_draw)
    cv.waitKey(0)

if __name__ == "__main__":
    points = []
    annotate = False
    q1_a()
    q1_b()