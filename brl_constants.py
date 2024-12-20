import pathlib
import os

### Task parameters
DATA_DIR = '/mnt/data1/act'
UPI_DATA_DIR = '/mnt/data1/upi'
TASK_CONFIGS = {
    'act_demo_z1_push_red':{
        'dataset_dir': DATA_DIR + '/act_demo_z1_push_red',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['wrist']
    },
    'act_demo_z1_push_random_red':{ 
        'dataset_dir': DATA_DIR + '/act_demo_z1_push_random_red',
        'num_episodes': 50,
        'episode_len': 200,
        'camera_names': ['wrist']
    },
    'act_demo_real_z1_push_sb':{
        'dataset_dir': DATA_DIR + '/act_demo_real_z1_push_sb',
        'num_episodes': 50,
        'episode_len': 900,
        'camera_names': ['wrist']
    },
    'act_demo_scara':{
        'dataset_dir': DATA_DIR + '/act_demo_scara_whiteboard_pick_one',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': ['wrist']
    },
    'act_demo_scara_infra':{
        'dataset_dir': DATA_DIR + '/act_demo_scara_whiteboard_pick_one',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': ['wrist', 'wrist_infra2']
    },
    'act_demo_scara_up_down_cam':{
        'dataset_dir': DATA_DIR + '/act_demo_scara_whiteboard_updown_pick_one',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': ['wrist', 'wrist_down']
    },
    'act_demo_scara_leaf_block':{
        'dataset_dir': DATA_DIR + '/act_demo_scara_whiteboard_pick_under_leaf',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
    'act_demo_scara_fruit_block':{
        'dataset_dir': DATA_DIR + '/train-act-scara-fruit-block',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
    'act_demo_scara_one_fruit_block':{
        'dataset_dir': DATA_DIR + '/train_act_scara_pick_onefruit_block',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
    'act_demo_scara_one_fruit_block_pick_cotrain':{
        'dataset_dir': DATA_DIR + '/train_act_scara_pick_onefruit_block_pick_data_cotrain',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
    'act_demo_scara_realfruit_block':{
        'dataset_dir': DATA_DIR + '/train_act_scara_pick_realfruit_block',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
    'act_scara_shunba':{
        'dataset_dir': DATA_DIR + '/train_act_scara_shunba',
        'num_episodes': 66,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_shunba_finetune_block1':{
        'dataset_dir': DATA_DIR + '/train_act_scara_shunba_finetune_block1',
        'num_episodes': 66,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_shunba_finetune_block2':{
        'dataset_dir': DATA_DIR + '/train_act_scara_shunba_finetune_block2',
        'num_episodes': 66,
        'episode_len': 500,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_mag_fruit_block':{
        'dataset_dir': DATA_DIR + '/train_act_scara_mag_fruit_block',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_3cam':{
        'dataset_dir': DATA_DIR + '/train_act_scara_3cam',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_down', 'top']
    },
        'act_scara_2cam':{
        'dataset_dir': DATA_DIR + '/train_act_scara_2cam',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_2cam':{
        'dataset_dir': DATA_DIR + '/train_act_scara_2cam',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_sim_env':{
        'dataset_dir': DATA_DIR + '/train_act_scara_sim-env-block-pick',
        'num_episodes': 400,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_down']
    },
        'act_scara_sim_env_wrist':{
        'dataset_dir': DATA_DIR + '/train_act_scara_sim-env-block-pick',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist']
    },
        'act_scara_sim_env_wrist_down':{
        'dataset_dir': DATA_DIR + '/train_act_scara_sim-env-block-pick',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist_down']
    },
        'act_segaug_scara_sim_env':{
        'dataset_dir': DATA_DIR + '/train_act_scara_sim-env-block-pick',
        'num_episodes': 400,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_down', 'wrist_segaug', 'wrist_down_segaug']
    },
        'act_segaug_scara_sim_env_wrist':{
        'dataset_dir': DATA_DIR + '/train_act_scara_sim-env-block-pick',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist', 'wrist_segaug']
    },
        'act_segaug_scara_sim_env_wrist_down':{
        'dataset_dir': DATA_DIR + '/train_act_scara_sim-env-block-pick',
        'num_episodes': 100,
        'episode_len': 300,
        'camera_names': ['wrist_down', 'wrist_down_segaug']
    },
        'upi_actep':{
        'dataset_dir': UPI_DATA_DIR + '/lab_sb/rs435i_lab_picking_024-08-29-09-07-12/grip_data_seg',
        'num_episodes': 100,
        'episode_len': 100,
        'camera_names': ['color']
    },
}
### Z1 envs fixed constants
DT = 0.033333
FPS = 30
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
