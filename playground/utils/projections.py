"""
--------------------------------------
Functions for camera image projection
--------------------------------------
Zhenghao Fei

Camera optical coordination is defined by opencv&ROS:
+z points to the front of the camera
+x points to the right of the camera
+y points to the down of the camera
"""
import numpy as np
import transformation as trans
import cv2 as cv

def project_line_to_image(line, projection_matrix):
    """
    line: [[x, y, z], [x, y, z]]
    projection_matrix: 3*4 camera projection matrix
    """
    return line_on_image

def project_point_to_image(points, projection_matrix):
    """
    point: N*[x, y, z] in optical frame
    projection_matrix: 3*4 camera projection matrix
    """
    pt_homo = trans.xyz2homo(points).T
    point_on_image = (projection_matrix.dot(pt_homo)/pt_homo[2, :]).T[:, :2]
    return point_on_image


def cutoff_line(line_pts, cutoff_z=1):
    """
    if a line has only one point infront of the camera, 
    find the intersect point to z plane in front of the camera,
    if both point are behind the cutoff_z, return valid = False
    [In] line_pts 4*2: [[x1, y1, z1], 
                        [x2, y2, z2]]
         [cutoff_z = 1] meter
    [out] valid, [line_pt0, line_pt1]     
    """
    pt0 = np.asarray(line_pts[:, 0])
    pt1 = np.asarray(line_pts[:, 1])

    valid = True
    
    #  both points are in the front, no need to do anything.
    if pt0[2] >=cutoff_z and pt1[2] >= cutoff_z:
        return valid, np.asarray([pt0, pt1]).T
    
    #  both points are in the back, invalid
    if pt0[2] <=cutoff_z and pt1[2] <= cutoff_z:
        valid = False
        
    if valid:
        if pt1[2] <= cutoff_z:
            delta_vector = pt1 - pt0
            line_length = (cutoff_z - pt0[2])/delta_vector[2]
            pt1 = pt0 + line_length*delta_vector
        else:
            delta_vector = pt0 - pt1
            line_length = (cutoff_z - pt1[2])/delta_vector[2]
            pt0 = pt1 + line_length*delta_vector  
        
    return valid, np.asarray([pt0, pt1]).T

def draw_mask_in_img(label_image, pts_in_image, color=(0, 0, 255), radius=2):
    for pt_in_image in pts_in_image:
        x, y = pt_in_image
        if x > 0 and x < label_image.shape[1] and y > 0 and y < label_image.shape[0]:
            label_image = cv.circle(label_image, (int(x), int(y)), radius, color)
    return label_image