import PyKDL as kdl
import kdl_parser_py.urdf
import numpy as np
import matplotlib.pyplot as plt
import transformation as trans
import cv2 as cv
import projections as proj

def joint_states_to_jnt_array(joint_states):
    # Convert joint states to a KDL JntArray
    kdl_joint_array = kdl.JntArray(len(joint_states))
    for i, value in enumerate(joint_states):
        kdl_joint_array[i] = value
    kdl_joint_array
    return kdl_joint_array

def get_frame_pose(fk_solver, kdl_joint_array):
    frame = kdl.Frame()
    fk_solver.JntToCart(kdl_joint_array, frame)
    # Extract position and orientation
    position = [frame.p.x(), frame.p.y(), frame.p.z()]
    orientation = [frame.M.GetQuaternion()[0], frame.M.GetQuaternion()[1],
                    frame.M.GetQuaternion()[2], frame.M.GetQuaternion()[3]]
    return position, orientation

def joint_states_to_ee_pose(all_joint_states, fk_solver):
    positions = []
    orientations = []
    for i, joint_states in enumerate(all_joint_states):
        kdl_joint_array = joint_states_to_jnt_array(joint_states)
        position, orientation = get_frame_pose(fk_solver, kdl_joint_array)
        positions.append(position)
        orientations.append(orientation)
        
    positions = np.asarray(positions)
    orientations = np.asarray(orientations)
    return positions, orientations

def plot_3d_trajectory(traj, figsize=(8, 6), title="EE Trajectory"):
    """
    Plots a 3D trajectory with enhanced visualization.

    Args:
        traj (list of lists): The 3D trajectory data, where each inner list represents a point (x, y, z).
        figsize (tuple, optional): Figure size in inches. Defaults to (8, 6).
        title (str, optional): Plot title. Defaults to "3D Trajectory".
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory with clear markers and line
    ax.plot(*zip(*traj), color='blue', marker='o', linestyle='-', linewidth=1, markersize=1)

    # Set labels, title, and initial viewpoint
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(-1, +1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 0.5)
    ax.set_title(title)
    ax.view_init(elev=20, azim=-60)  # Adjust viewpoint as needed

    # Optional: Add grid and axes limits
    ax.grid(True)
    # Set appropriate limits based on your data range

    plt.show()

def get_poses(fk_solver, chain, joint_states):
    # Loop over all segments in the chain
    poses = []
    for i in range(chain.getNrOfSegments()):
        # Compute the Cartesian pose for the current joint state
        pose = kdl.Frame()
        fk_solver.JntToCart(joint_states, pose, i+1)
        segment_name = chain.getSegment(i).getName()
        poses.append((segment_name, pose))
        # print(f"Name: {segment_name}, Pose: \n {pose}")
    return poses

def plot_robot(poses):
    """
    Plot the robot in 3D space using matplotlib.
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Loop over all poses
    prev_x, prev_y, prev_z = None, None, None
    for name, pose in poses:
        # Extract the x, y, and z coordinates from the pose
        x, y, z = pose.p[0], pose.p[1], pose.p[2]
        
        # Plot the point on the axes
        ax.scatter(x, y, z)
        
        # If it's not the first point, draw a line from the previous point to the current point
        if prev_x is not None:
            ax.plot([prev_x, x], [prev_y, y], [prev_z, z])
        
        prev_x, prev_y, prev_z = x, y, z
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 0.6])
    ax.set_xlim([0, 0.6])

    # axis label
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Display the plot
    # plt.show()
    return ax


def plot_robot_2d_xy(poses):
    """
    Plot the robot in 2d x-y space using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Loop over all poses
    prev_x, prev_y, prev_z = None, None, None
    for name, pose in poses:
        # Extract the x, y, and z coordinates from the pose
        x, y, z = pose.p[0], pose.p[1], pose.p[2]
        
        # Plot the point on the axes
        ax.scatter(x, y)
        
        # If it's not the first point, draw a line from the previous point to the current point
        if prev_x is not None:
            ax.plot([prev_x, x], [prev_y, y])
        prev_x, prev_y, prev_z = x, y, z
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([-0.01, 1])

    # axis label
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Display the plot
    plt.show()

def get_colormap_color(value, colormap_code=cv.COLORMAP_JET):
  """
  Maps a value to a color using the specified colormap.

  Args:
      value (float): Value to be mapped (0.0 to 1.0).
      colormap_code (int, optional): OpenCV colormap code. Defaults to cv.COLORMAP_JET.

  Returns:
      tuple: RGB color tuple.
  """
  cmap = cv.applyColorMap(np.array([[value * 255]], dtype=np.uint8), colormap_code)
  return tuple(cmap[0][0])

def plot_robot_2d_xy_batch(link_poses_list):
    """
    Plot the robot in 2d x-y space using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, link_poses in enumerate(link_poses_list):
        # Loop over all poses
        prev_x, prev_y, prev_z = None, None, None
        for name, pose in link_poses:
            # Extract the x, y, and z coordinates from the pose
            x, y, z = pose.p[0], pose.p[1], pose.p[2]
            
            # Plot the point on the axes
            ax.scatter(x, y, c='r')
            
            # If it's not the first point, draw a line from the previous point to the current point
            if prev_x is not None:
                ax.plot([prev_x, x], [prev_y, y], c='r')
            prev_x, prev_y, prev_z = x, y, z
    ax.set_ylim([-0.5, 0.1])
    ax.set_xlim([-0.01, 0.6])

    # axis label
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

def get_camera_poses(vis_actions, fk_solver, chain):
    link_poses_list = []
    ee_poses = []
    ee_poses_raw = []
    for i in range(0, len(vis_actions), 1):
        vis_action = vis_actions[i]
        joint_states = joint_states_to_jnt_array(vis_action[:4])
        link_poses = get_poses(fk_solver, chain, joint_states)
        link_poses_list.append(link_poses)
        ee_poses.append([link_poses[-1][1].p[0], link_poses[-1][1].p[1], link_poses[-1][1].p[2]])
        ee_poses_raw.append(link_poses[-1][1])
    ee_poses = np.asarray(ee_poses)
    return ee_poses, ee_poses_raw, link_poses_list

def get_camera_in_world_and_init(ee_poses_raw):
    transformation_matrix, rot_mat_ext, translation_mat= trans.kdl_frame_to_mat(ee_poses_raw[0])
    ee_in_world_init = transformation_matrix
    world_T_init = np.linalg.inv(ee_in_world_init)

    ee_in_world_all = []
    ee_in_init_all = []
    ee_in_init_all_pos = []
    ee_in_world_all_pos = []
    for ee_pose in ee_poses_raw:
        ee_in_world_all_pos.append([ee_pose.p[0], ee_pose.p[1], ee_pose.p[2], 1])
        ee_in_world, _, _= trans.kdl_frame_to_mat(ee_pose)
        ee_in_world_all.append(ee_in_world)
        ee_in_init = world_T_init.dot(ee_in_world)
        ee_in_init_all.append(ee_in_init)
        ee_in_init_all_pos.append(ee_in_init[:3, 3])
        
        # print("ee rpy: ", tf.euler_from_matrix(ee_in_init, 'rxyz'))

    ee_in_init_all_pos = np.asarray(ee_in_init_all_pos)
    ee_in_world_all_pos = np.asarray(ee_in_world_all_pos)
    return  ee_in_init_all_pos, ee_in_world_all_pos

def pts_cam_to_opt(ee_pts_cam):
    ee_pts_opt = ee_pts_cam.copy()
    ee_pts_opt = ee_pts_opt[:, [1, 2, 0]]
    ee_pts_opt[:, 2] = ee_pts_opt[:, 2]
    ee_pts_opt[:, 0] = -ee_pts_opt[:, 0]
    ee_pts_opt[:, 1] = -ee_pts_opt[:, 1]
    return ee_pts_opt

def paint_action_in_image(img_plot, vis_actions, proj_mat, fk_solver, chain, save_path=None, plt_fig_ax=None, title=""):
    if plt_fig_ax is None:
        fig, ax = plt.subplots()  # Create a single figure and axis
    else:
        fig, ax = plt_fig_ax

    ax.cla()
    cam_poses, cam_poses_raw, link_poses_list = get_camera_poses(vis_actions, fk_solver, chain)
    cam_pts_init, cam_in_world_all_pos = get_camera_in_world_and_init(cam_poses_raw)
    # cam_pts_init to ee_pts_cam (ee_pts_init, init is the same as cam)
    camera_T_ee = trans.states2SE3([0.12, -0.008, -0.0485, 0, 0, 0])
    ee_pts_cam = camera_T_ee.dot(trans.xyz2homo(cam_pts_init).T).T[:,: 3]
    ee_pts_opt = pts_cam_to_opt(ee_pts_cam)
    uvs = proj.project_point_to_image(ee_pts_opt, proj_mat)
    ax.imshow(cv.cvtColor(img_plot, cv.COLOR_BGR2RGB))
    ax.scatter(uvs[:, 0], uvs[:, 1], c=range(len(uvs)), s=10)
    # limit the axis to the image shape
    ax.set_xlim([0, img_plot.shape[1]])
    ax.set_ylim([img_plot.shape[0], 0])
    # turn off the axis
    ax.axis('off')
    # ax.colorbar()
    # save the image
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, format='jpg')
    else:
        fig.show()
