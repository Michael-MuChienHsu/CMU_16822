import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from q2_utils import *

def click_event_line(event, x, y, flags, params):
    """ Directly draw line on global param: label_img"""
    if event == cv.EVENT_LBUTTONDOWN: 
        print(x, y)
        points.append( [x, y] )
        cv.circle( label_img, (x, y), radius=2, color=get_color(1), thickness=-1 )
        if len(points) % 2 == 0:
            cv.line(label_img, points[-2], points[-1], get_color(2), thickness=5)

        cv.imshow('label_img', label_img)

def click_event_plane(event, x, y, flags, params):
    """ Directly draw square on global param: label_img"""
    color_code = len(points)//4
    if event == cv.EVENT_LBUTTONDOWN: 
        points.append( [x, y] )
        cv.circle( label_img, (x, y), radius=2, color=get_color(color_code), thickness=-1 )
        cv.putText(label_img, str(len(points)-1), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, get_color(color_code), 2)
        if len(points) % 4 == 0:
            # cv.line(label_img, points[-2], points[-1], get_color(color_code), thickness=5)
            cv.polylines(label_img, pts = [np.array(points[-4:])], isClosed = True, color = get_color(color_code), thickness=4)

        cv.imshow('label_img', label_img)

def add_mouse_response(img_name, reset = False, click_event = click_event_line):
    """Catch mouse response event on img_name."""

    global points
    if reset:
        points = []

    cv.setMouseCallback(img_name, click_event) 
    cv.waitKey(0)
    cv.destroyAllWindows()

def q3(subsample):
    global points
    points = []
    global label_img

    image_path = "./data/q3.png"
    image = cv.imread(image_path)
    cv.imwrite("q3_origin_image.png", image)

    # step 1: Annotate 3 parallel lines and use q2a compute K.
    if annotate:
        label_img = image.copy()
        cv.imshow( "label_img", label_img )
        print("Annotate points in clockwise.")
        add_mouse_response("label_img")
    else:
        points = np.load("./data/q3/q3.npy")
        points = [points[0][0], points[0][1], points[0][3], points[0][2], 
                  points[1][1], points[1][0], points[1][2], points[1][3],
                  points[0][0], points[0][3], points[1][1], points[2][2],]

    points = [list(p) for p in points]
    lines = points_to_lines(points)
    vanishing_points = []
    for i in range(lines.shape[0]//2):
        intersect = get_intersection( lines[2*i], lines[2*i+1] ) 
        vanishing_points.append( intersect )

    K = get_K(vanishing_points)
    print("Q3 K is:")
    print(K)

    # step 2: Label 5 planes
    if annotate:
        label_img = image.copy()
        cv.imshow( "label_img", label_img )
        print("Annotate points in clockwise.")
        add_mouse_response("label_img", reset=True, click_event=click_event_plane)
        print(points)
    else:
        # Use the provided points for to reproduce.
        points = np.load("./data/q3/q3.npy").ravel().astype(int).reshape(-1, 2)

    # Draw annotated planes.
    anno_image = image.copy()
    plane_corners = [ points[4*i:4*i+4] for i in range(len(points)//4) ]
    for i, plane in enumerate(plane_corners):
        cv.polylines(anno_image, pts = [np.array(plane)], isClosed = True, color = get_color(i), thickness=4)
        for j, p in enumerate(plane):
            cv.putText(anno_image, str(i*4+j), tuple(p), cv.FONT_HERSHEY_SIMPLEX, 1, get_color(i), 2)
    cv.imwrite( "q3_labeled_image.png", anno_image )

    # Step 3: Compute plane normals
    homo_planes = [ [ [p[0], p[1], 1] for p in plane] for plane in plane_corners ]
    plane_orientations = []
    for plane in homo_planes:
        plane_orientations.append(get_plane_noraml(plane, K))

    # step 4: For each plane compute the ray for each pixel in the plane
    # step 4.1 Identify coordinates for the pixel in the plane
    all_plane_coor = [] # list of np.array
    for contour in plane_corners:
        contour = np.array(contour)
        homo_coor = []
        x, y, w, h = cv.boundingRect(contour)
        for _y in range(y, y+h):
            for _x in range(x, x+w):
                if cv.pointPolygonTest(contour, (_x, _y), False ):
                    homo_coor.append( [_x, _y, 1] )

        all_plane_coor.append(np.array(homo_coor))

    all_plane_ray = []
    for homo_coor in all_plane_coor:
        homo_coor = np.array(homo_coor)
        all_plane_ray.append( np.linalg.inv(K).dot(homo_coor.T).T )
    
    # Pick a ray for reference ray to set depth.
    reference_point = np.array([514, 401, 1])
    reference_ray = np.linalg.inv(K).dot(reference_point)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # For each plane perform step 5 and 6.
    # print(image.shape)
    for i, ( _plane_coor, _plane_ray, _orientation  ) in enumerate(zip(all_plane_coor, all_plane_ray, plane_orientations)):
        _plane_coor = _plane_coor[:, :2]
        # print( _plane_ray.shape )
        if subsample > 0:
            if type(subsample) == int:
                subsample_size = subsample
                print(f"Constant subsample_size = {subsample_size}")

            else:
                subsample_size = int( len(_plane_coor)*subsample )
                print(f"sampling ratio = {subsample}, subsample_size = {subsample_size}")

            sampled_idx = np.random.choice(len(_plane_coor), size=subsample_size, replace=False)
            _plane_coor, _plane_ray= _plane_coor[sampled_idx], _plane_ray[sampled_idx]
            
        print(f"Working on plane {i}")

        # Step 5 Compute plane equation.
        a = -_orientation.dot(reference_ray)

        # Step 6 Compute 3D coordinate.
        _positin_3d = -a*_plane_ray/(_plane_ray.dot(np.array(_orientation))[:, None]) 
        x_pos = _plane_coor.T[0]
        y_pos = _plane_coor.T[1]
        color_list = image[tuple(y_pos), tuple(x_pos)]/255
        color_list = [tuple(c[::-1]) for c in color_list]
        x_3d, y_3d, z_3d = _positin_3d.T
        ax.scatter( x_3d, y_3d, z_3d, color = color_list, s=0.1 )
    ax.view_init(90, 45)  
    plt.show()

if __name__ == "__main__":
    annotate = False
    subsample = -1
    q3(subsample)