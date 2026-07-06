from gym import spaces
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from suite.act.constants import DT, XML_DIR, START_ARM_POSE
from suite.act.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from suite.act.constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from suite.act.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from suite.act.constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

BOX_POSE = [None]                              


def make_sim_env(task_name, speedup=False):
    if "sim_transfer_cube" in task_name:
        if not speedup:
            print("Using normal gain gripper")
            xml_path = os.path.join(XML_DIR, f"bimanual_viperx_transfer_cube.xml")
        else:
            print("Using high gain gripper")
            xml_path = os.path.join(XML_DIR, f"bimanual_viperx_transfer_cube_high_gain.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(
            physics,
            task,
            time_limit=20,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    elif "sim_insertion" in task_name:
        if not speedup:
            xml_path = os.path.join(XML_DIR, f"bimanual_viperx_insertion.xml")
        else:
            xml_path = os.path.join(XML_DIR, f"bimanual_viperx_insertion_high_gain.xml")
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(
            physics,
            task,
            time_limit=20,
            control_timestep=DT,
            n_sub_steps=None,
            flat_observation=False,
        )
    else:
        raise NotImplementedError

    env.action_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
    env.observation_space = spaces.Dict(
        {
            "qpos": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
            "qvel": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
            "images": spaces.Dict(
                {
                    "top": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    "angle": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    "vis": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                }
            ),
        }
    )
    env.metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}
    env.seed = lambda x: x                       
    return env


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_left_gripper_action
        )
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
            normalized_right_gripper_action
        )

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [
                left_arm_action,
                full_left_gripper_action,
                right_arm_action,
                full_right_gripper_action,
            ]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate(
            [left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos]
        )

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate(
            [left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel]
        )

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = dict()
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(
            height=480, width=640, camera_id="angle"
        )
        obs["images"]["vis"] = physics.render(
            height=480, width=640, camera_id="front_close"
        )

        return obs

    def get_reward(self, physics):
                                                        
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
                                              
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
                                   
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
                                                        
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = (
            "red_box",
            "vx300s_left/10_left_gripper_finger",
        ) in all_contact_pairs
        touch_right_gripper = (
            "red_box",
            "vx300s_right/10_right_gripper_finger",
        ) in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:          
            reward = 2
        if touch_left_gripper:                      
            reward = 3
        if touch_left_gripper and not touch_table:                       
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
                                              
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]               
                                   
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
                                            
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = (
            "red_peg",
            "vx300s_right/10_right_gripper_finger",
        ) in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:              
            reward = 1
        if (
            touch_left_gripper
            and touch_right_gripper
            and (not peg_touch_table)
            and (not socket_touch_table)
        ):              
            reward = 2
        if (
            peg_touch_socket and (not peg_touch_table) and (not socket_touch_table)
        ):                           
            reward = 3
        if pin_touched:                        
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
                
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
                    
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7 + 6] = normalized_right_pos
    return action
