

import numpy as np
import os, sys
import torch
import cv2
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
from utils import flatten_list, BatchSampler
import fnmatch


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

def find_all_datafolder(dataset_dir):
    datafolders = []
    for root, dirs, files in os.walk(dataset_dir, followlinks=True):
        for dirname in fnmatch.filter(dirs, 'seg_*'):
            datafolders.append(os.path.join(root, dirname))
    print(f'Found {len(datafolders)} datafolders under {dataset_dir}')
    return datafolders

def get_norm_stats_folder(dataset_path_list):
    all_pose_data = []
    all_grip_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            action = np.loadtxt(dataset_path + "/gripper_distances.txt").reshape(-1, 1)
            camera_trajs = np.loadtxt(dataset_path + "/CameraTrajectory.txt")
            camera_trajs_mat = camera_trajs.reshape(-1, 3, 4)
            pose = np.zeros((len(camera_trajs), 6))
            pose[:, :3] = camera_trajs_mat[:, :3, 3]
            euler_angles = rotation_matrix_to_euler_angles(camera_trajs_mat[:, :3, :3])
            pose[:, 3:] = euler_angles
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_pose_data.append(torch.from_numpy(pose))
        all_grip_data.append(torch.from_numpy(action))
        all_episode_len.append(len(action))
    all_pose_data = torch.cat(all_pose_data, dim=0)
    all_grip_data = torch.cat(all_grip_data, dim=0)
    # cat pose and grip data
    all_action_data = torch.cat([all_pose_data, all_grip_data], dim=1)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps}

    return stats, all_episode_len


def load_folderdata(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99, resize=None, seed=0, fix_val=False):
    np.random.seed(seed)

    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_datafolder(dataset_dir) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    print("dataset_path_list_list: ", dataset_path_list_list)

    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)
    val_episode_ids_0 = []
    train_episode_ids_0 = []
    # val episodes are pathname contain "_val"
    if fix_val:
        print("Fix validation set on datafolders with _val")
        for idx, dataset_path in enumerate(dataset_path_list):
            if "_val" in dataset_path:
                val_episode_ids_0.append(idx)
            else:
                train_episode_ids_0.append(idx)
    else:
        print("Random validation set")
        shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
        train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
        val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]

    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    print("train_episode_ids_l: ", train_episode_ids_l)
    print("val_episode_ids_l: ", val_episode_ids_l)

    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats_folder(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats_folder(flatten_list([find_all_datafolder(stats_dir) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')
    # for id in train_episode_ids:
    #     print("train data: ", dataset_path_list[id])
    for id in val_episode_ids:
        print("val data: ", dataset_path_list[id])
    # construct dataset and dataloader
    train_dataset = EpisodicDatasetRs435i(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, resize=resize)
    val_dataset = EpisodicDatasetRs435i(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class, resize=resize)
    train_num_workers = (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 8
    val_num_workers = 8 if train_dataset.augment_images else 8
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

