import PyKDL as kdl
import kdl_parser_py.urdf
import numpy as np
import matplotlib.pyplot as plt


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
    fig = plt.figure()
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
    plt.show()


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
