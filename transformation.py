import numpy as np
import math
import transformations as tf

#-----------------------
# Transformation
#-----------------------
def eulerAnglesToRotationMatrix(theta):
    # Calculates Rotation Matrix given euler angles.

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

def isRotationMatrix(R) :
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R) :
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    assert(isRotationMatrix(R))    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def xyz2homo(xyz):
    # xyz: n*3
    # homo: n*4
    homo = np.concatenate([xyz, np.ones([len(xyz), 1])], axis=1)
    return homo

def normalizeHomo(homos):
    p = homos[:, -1]
    p = np.expand_dims(p, axis=1)
    normailzed_homos = homos/p
    return normailzed_homos


def states2SE3(states):
    # 6d states x, y, z, r, p ,yaw
    x, y, z, roll, pitch, yaw = states
    transformation = np.diag([1., 1., 1., 1.])
    rotation_matrix = eulerAnglesToRotationMatrix([roll, pitch, yaw])
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = [x, y, z]
    return transformation

def SE32states(trans):
    states = np.zeros(6)
    eulers = rotationMatrixToEulerAngles(trans[:3, :3])
    states[3:] = eulers
    states[:3] = trans[:3, 3]
    return states

def transFromTwoPoses(pose1, pose2, SE3=False):
    pose1_se3 = states2SE3(pose1)
    pose2_se3 = states2SE3(pose2)
    trans12_se3 = np.dot(np.linalg.inv(pose1_se3), pose2_se3)
    if SE3:
        return trans12_se3
    else:
        trans12_6d = SE32states(trans12_se3)
        return trans12_6d

def transformPoints(points, trans_6d):
    # In: [1] points: N*3
    # Out: [1] points: N*3
    points = xyz2homo(points).T
    trans_matrix = states2SE3(trans_6d)
    points = (np.dot(trans_matrix, points)).T
    points = normalizeHomo(points)
    points = points[:, :3]
    return points
    
def transform2DPoints(points, trans):
    # In: [1] points: N*2
    # Out: [1] points: N*2
    points =  np.concatenate([points, np.zeros([len(points), 1])], axis=1)
    points = xyz2homo(points).T
    if len(trans) == 6:
        trans_matrix = states2SE3(trans)
    else:
        trans_matrix = trans

    points = (np.dot(trans_matrix, points)).T
    points = normalizeHomo(points)
    points = points[:, :2]
    return points
    
def vector_quat_to_matrix(vector, quat):
    """
    Converts a vector and a quaternion to a transformation matrix.
    input:
        vector: a 3-dimensional vector (x, y, z)
        quat: a 4-dimensional quaternion (x, y, z, w)
    output:
        transformation_matrix: a 4x4 transformation matrix
    """
    rotation_matrix = tf.quaternion_matrix([quat[3], quat[0], quat[1], quat[2]])
    translation_matrix = tf.translation_matrix(vector)
    transformation_matrix = translation_matrix.dot(rotation_matrix)
    return transformation_matrix

def kdl_frame_to_mat(frame):
    """
    Convert a PyKDL Frame to a 4x4 numpy matrix.
    """
    rot_mat = np.asarray([[frame.M[0, 0], frame.M[0, 1], frame.M[0, 2]],
                        [frame.M[1, 0], frame.M[1, 1], frame.M[1, 2]],
                        [frame.M[2, 0], frame.M[2, 1], frame.M[2, 2]]])
    rot_mat_ext = np.eye(4)
    rot_mat_ext[:3, :3] = rot_mat

    trans_vector = np.asarray([frame.p[0], frame.p[1], frame.p[2]])
    translation_mat = tf.translation_matrix(trans_vector)
    transformation_matrix = translation_mat.dot(rot_mat_ext)
    transformation_matrix
    return transformation_matrix, rot_mat_ext, translation_mat
