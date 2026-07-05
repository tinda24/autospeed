# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env

from sensor_msgs.msg import CompressedImage

import collections
from collections import deque
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def save_data(args, timesteps, actions, dataset_path):
    actions_data_size = len(actions)
    img_data_size = len(timesteps)
    print(f'action len:{actions_data_size}, img len: {img_data_size}')
    data_dict = {
        '/qpos': [],
        '/action': [],
    }

    for cam_name in args.camera_names:
        data_dict[f'/{cam_name}'] = []

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0) if len(timesteps) > 0 else None
        data_dict['/action'].append(action)
        if ts != None:
            data_dict['/qpos'].append(ts.observation['qpos'])
            for cam_name in args.camera_names:
                img = ts.observation['images'][cam_name]
                #region Resize
                img_new = cv2.resize(img,(406,336))
                data_dict[f'/{cam_name}'].append(img_new)


    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        for cam_name in args.camera_names:
            _ = root.create_dataset(cam_name, (img_data_size, 336, 406, 3), dtype='uint8',
                                         chunks=(1, 336, 406, 3), )
        _ = root.create_dataset('qpos', (img_data_size, 14))
        _ = root.create_dataset('action', (actions_data_size, 14))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)


class RosOperator:
    def __init__(self, args):

        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_global_deque = None

        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()

        print('RosOperator init well')

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_global_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_global_deque) == 0:
            return False
        
        frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(),self.img_global_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_global_deque) == 0 or self.img_global_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        # camera1
        while self.img_global_deque[0].header.stamp.to_sec() < frame_time:
            self.img_global_deque.popleft()
        img_global = self.img_global_deque.popleft()
        
        np_arr = np.frombuffer(img_global.data, np.uint8)
        img_global = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_global = img_global[:, :, ::-1]

        # camera2
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.img_left_deque.popleft()
        
        np_arr = np.frombuffer(img_left.data, np.uint8)
        img_left = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_left = img_left[:, :, ::-1]

        # camera3
        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.img_right_deque.popleft()
        
        np_arr = np.frombuffer(img_right.data, np.uint8)
        img_right = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_right = img_right[:, :, ::-1]

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        return (img_left, img_right,img_global,
                puppet_arm_left, puppet_arm_right)

    def get_actions_only(self):
        if len(self.puppet_arm_left_deque) == 0 or len(self.puppet_arm_right_deque) == 0:
            return False
        frame_time = min([self.puppet_arm_left_deque[-1].header.stamp.to_sec(), self.puppet_arm_right_deque[-1].header.stamp.to_sec()])

        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        
        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()
        return puppet_arm_left, puppet_arm_right

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)
    
    def img_global_callback(self, msg):
        if len(self.img_global_deque) >= 2000:
            self.img_global_deque.popleft()
        self.img_global_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, CompressedImage, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, CompressedImage, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_global_topic, CompressedImage, self.img_global_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):
        timesteps = []
        actions = []
        image = np.random.randint(0, 255, size=(336, 406, 3), dtype=np.uint8)
        image_dict = dict()
        for cam_name in self.args.camera_names:
            image_dict[cam_name] = image
        count = 0

        actions_rate_ratio = self.args.frame_actions_ratio
        rate = rospy.Rate(self.args.frame_rate * actions_rate_ratio) 
        print_flag = True

        while (count < self.args.max_timesteps + 1) and not rospy.is_shutdown():
            # print(count)
            if count % actions_rate_ratio == 0: # get images and actions
                result = self.get_frame()
                if not result:
                    if print_flag:
                        print("syn fail")
                        print_flag = False
                    rate.sleep()
                    continue
                print_flag = True
                count += 1
                (img_left, img_right, img_global,
                puppet_arm_left, puppet_arm_right) = result
                
                image_dict = dict()
                image_dict[self.args.camera_names[0]] = img_left
                image_dict[self.args.camera_names[1]] = img_right
                image_dict[self.args.camera_names[2]] = img_global

                obs = collections.OrderedDict()
                obs['images'] = image_dict
                obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)

                if count == 1:
                    ts = dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=obs)
                    timesteps.append(ts)
                    continue

                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.MID,
                    reward=None,
                    discount=None,
                    observation=obs)

                action = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
                actions.append(action)
                timesteps.append(ts)
                # print(f'count:{count}, save images and actions')
                print("Frame data: ", count)
                if rospy.is_shutdown():
                    exit(-1)
                rate.sleep()
            else:  # only get actions
                result = self.get_actions_only()
                if not result:
                    if print_flag:
                        print("syn fail")
                        print_flag = False
                    rate.sleep()
                    continue
                print_flag = True
                count += 1
                puppet_arm_left, puppet_arm_right = result
                if count == 1:
                    ts = dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=obs)
                    timesteps.append(ts)
                    continue

                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.MID,
                    reward=None,
                    discount=None,
                    observation=obs)

                action = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
                actions.append(action)
                print("Frame data: ", count)
                if rospy.is_shutdown():
                    exit(-1)
                rate.sleep()
        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=200, required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=[ 'cam_left_wrist', 'cam_right_wrist','cam_global'], required=False)
    #  topic name of color image
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera3/color/image_raw/compressed', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera2/color/image_raw/compressed', required=False)
    parser.add_argument('--img_global_topic', action='store', type=str, help='img_global_topic',
                        default='/camera1/color/image_raw/compressed', required=False)
    # topic name of arm
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--frame_actions_ratio', action='store', type=int, help='frame_rate',
                        default=2, required=False)
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    timesteps, actions = ros_operator.process()
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    
    if(len(actions) < args.max_timesteps):
        print("\033[31m\nSave failure, please record %s timesteps of data.\033[0m\n" %args.max_timesteps)
        exit(-1)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, "episode_" + str(args.episode_idx))
    save_data(args, timesteps, actions, dataset_path)


if __name__ == '__main__':
    main()