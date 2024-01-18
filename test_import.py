import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import time
from torchvision import transforms
import h5py
import cv2 as cv
import sys

act_path = "/codes/act-plus-plus/"
sys.path.append(act_path)
from brl_constants import FPS
from brl_constants import PUPPET_GRIPPER_JOINT_OPEN
from brl_constants import TASK_CONFIGS
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import (
    compute_dict_mean,
    set_seed,
    detach_dict,
    calibrate_linear_vel,
    postprocess_base_action,
)  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
import argparse

print("All import success")

print("Test CUDA")
print(torch.cuda.is_available())
