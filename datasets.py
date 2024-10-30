

import numpy as np
import os, sys
import torch
import cv2
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation


class EpisodicDatasetRs435i(torch.utils.data.Dataset):
    """
    Custom dataset, collected from UPI, using RS D435i camera
    Dataset structure:
    - dataroot (rs435i)
    -- seg_0000x
    --- camera_info_color.txt
    --- camera_info_left.txt
    --- camera_info_right.txt
    --- CameraTrajectory.txt
    --- gripper_distances.txt
    --- times.txt
    --- color_00000x.png
    --- right_00000x.png
    --- left_00000x.png

    For each timestep:
    observations
    - images
        - each_cam_name     (480, 848, 3) 'uint8'
        - each_infra_name (480, 848, 3) 'uint8' (use 3 channel for purpose of data compatibility)
        - each_depth_name     (480, 848, 1) 'uint16' (not in use yet)
    - states
        - action               (7)         'float32' - x, y, z, r, p, y, gripper

    Zhenghao Fei
    """
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, resize=None):
        super(EpisodicDatasetRs435i).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.resize = resize
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False

        print("augment_images: ", self.augment_images)
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        print("episode_id: ", episode_id)
        print("start_ts: ", start_ts)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            is_sim = False
            grip = np.loadtxt(dataset_path + "/gripper_distances.txt").reshape(-1, 1)
            camera_trajs = np.loadtxt(dataset_path + "/CameraTrajectory.txt")
            camera_trajs_mat = camera_trajs.reshape(-1, 3, 4)
            # append [0, 0, 0, 1] to the end of each camera pose
            camera_trajs_mat = np.concatenate([camera_trajs_mat, np.tile(np.array([[0, 0, 0, 1]]), (camera_trajs_mat.shape[0], 1, 1))], axis=1)


            # transfer trajectory to relative pose (current pose as the origin)
            current_pose_init = camera_trajs_mat[start_ts]
            cam_T_init = np.linalg.inv(current_pose_init)
            camera_trajs_mat_in_current = np.matmul(cam_T_init, camera_trajs_mat)
            poses = transformation_matrix_to_pose(camera_trajs_mat_in_current)
            action = np.concatenate([poses, grip], axis=1)

            original_action_shape = action.shape
            episode_len = original_action_shape[0]

            # get observation at start_ts only
            image_dict = dict()
            for cam_name in self.camera_names:
                color_image_path = dataset_path + f"/{cam_name}_%06d.png" % start_ts
                image_dict[cam_name] = cv2.imread(color_image_path)
                if self.resize is not None:
                    image_dict[cam_name] = cv2.resize(image_dict[cam_name], (self.resize[0], self.resize[1]), interpolation=cv2.INTER_LINEAR)

            # get all actions after and including start_ts
            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

            # self.is_sim = is_sim
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action += action[-1] # pad action with last action
            padded_action[:action_len] = action

            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1
            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]


            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                if len(image_dict[cam_name].shape) == 2:
                    gary_to_rgb = np.dstack([image_dict[cam_name]] * 3)
                    all_cam_images.append(gary_to_rgb)
                else:
                    all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            
            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        except Exception as e:
            print(e)
            print(f'Error loading {dataset_path} in __getitem__')
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        return image_data, action_data, is_pad
    
def rotation_matrix_to_euler_angles(rotation_matrices):
    """Converts a batch of rotation matrices to Euler angles (roll, pitch, yaw).

    Args:
    rotation_matrices: A list of 3x3 rotation matrices.

    Returns:
    A list of Euler angles (roll, pitch, yaw) corresponding to each rotation matrix.
    """

    rotations = Rotation.from_matrix(rotation_matrices)
    euler_angles = rotations.as_euler('xyz', degrees=False)
    return euler_angles

def transformation_matrix_to_pose(transformation_matrices):
    """Converts a batch of transformation matrices to pose (x, y, z, roll, pitch, yaw).
    Args:
    transformation_matrices: A list of 4x4 transformation matrices.
    Returns:
    A list of pose (x, y, z, roll, pitch, yaw) corresponding to each transformation matrix.
    """
    pose = np.zeros((len(transformation_matrices), 6))
    pose[:, :3] = transformation_matrices[:, :3, 3]
    euler_angles = rotation_matrix_to_euler_angles(transformation_matrices[:, :3, :3])
    pose[:, 3:] = euler_angles
    return pose