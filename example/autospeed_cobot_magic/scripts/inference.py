import hydra
import yaml
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.environ.get("autospeed_ROOT", os.getcwd()), "repos"))
from einops import rearrange
import collections
from collections import deque
from sensor_msgs.msg import CompressedImage
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import threading
from pathlib import Path
import cv2
from einops import rearrange
import h5py
import sys
sys.path.append("./")
import sys
sys.path.append(os.getcwd())
from autospeed_cobot_magic.utils.nonlinear_temporal_agg import NonlinearTemporalAgg

task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None

def action_safety(action_left,action_right):
    limits_left_min = [-1.24,-0.0173,-2.65,-1.7,-1.283,-1.998,-0.0007]
    limits_left_max = [0.93,2.52,0.0226,1.662,1.25,2.0559,0.069]
    limits_right_min = [-1.09,-0.17496,-2.05,-1.74,-1.28,-2.1,0.0028]
    limits_right_max = [1.448,2.38,0.0265,1.739,1.26,2.08,0.0693]
    for i in range(7):
        left_min = limits_left_min[i]
        left_max = limits_left_max[i]
        right_min = limits_right_min[i]
        right_max = limits_right_max[i]
        action_left[i] = min(max(left_min,action_left[i]),left_max)
        action_right[i] = min(max(right_min,action_right[i]),right_max)
        
    return action_left,action_right


def get_image(image, new_size=(128, 128), rotate_180=False):
    image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC)
    
    if rotate_180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    
    image = rearrange(image, 'h w c -> c h w')
    
    image = image / 255.0
    
    return image

def inference_process(args, prompt, ros_operator, policy, t, pre_action,stats_dict):
    global inference_lock
    global inference_actions
    global inference_speed
    global inference_timestep
    print_flag = True
    rate = rospy.Rate(args.publish_rate)

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()

        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        obs = collections.OrderedDict()

        obs['cam_global'] = torch.from_numpy(get_image(img_front,args.img_size,rotate_180=False)).float().cuda()
        obs['cam_left_wrist'] = torch.from_numpy(get_image(img_left,args.img_size,rotate_180=True)).float().cuda()
        obs['cam_right_wrist'] = torch.from_numpy(get_image(img_right,args.img_size,rotate_180=True)).float().cuda()

        qpos = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        
        obs['proprioceptive'] = qpos
        start_time = time.time()
        
        with torch.no_grad():
            all_actions,pred_a = policy.act(obs, qpos, prompt, stats_dict)
            all_actions = all_actions.squeeze(0).cpu().numpy()
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions
        inference_speed = pred_a
        if pre_action is None:
            pre_action = obs['proprioceptive']
        inference_timestep = t
        inference_lock.release()
        break

def model_inference(args, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_speed
    global inference_timestep
    global inference_thread

    global inference_action_buffer
    global aggregator
    aggregator = NonlinearTemporalAgg()

    norm_dir = args.norm_dir
    stats_dict = {}       
    with h5py.File(norm_dir, 'r') as file:
        for key in file.keys():
            item = file[key]
            if isinstance(item, h5py.Group):
                for sub_key in item.keys():
                    sub_item = item[sub_key]
                    if isinstance(sub_item, h5py.Group):
                        for sub_sub_key in sub_item.keys():
                            sub_sub_item = sub_item[sub_sub_key]
                            if isinstance(sub_sub_item, h5py.Dataset):
                                stats_dict[f"{key}/{sub_key}/{sub_sub_key}"] = sub_sub_item[()]
                    elif isinstance(sub_item, h5py.Dataset):
                        stats_dict[f"{key}/{sub_key}"] = sub_item[()]
            elif isinstance(item, h5py.Dataset):
                stats_dict[key] = item[()]

    from omegaconf import OmegaConf
    config_path = os.path.join(args.config_dir)
    saved_yaml = OmegaConf.load(config_path)


    task_dist = saved_yaml['suite']['task']['tasks']
    assert len(task_dist)==1,f'task_dist len is {len(task_dist)}'
    task_dist = task_dist[0]
    task_dist = task_dist['desktop']
    print(task_dist)
    for i,task in enumerate(task_dist):
        print(f'{i}:  {task}')
    choosen_id = input('select the task id to exercute:')
    task_chosen = task_dist[int(choosen_id)]
    print(f'chosen task: {task_chosen}')

    from sentence_transformers import SentenceTransformer
    print('lang model loading')
    lang_model = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
    )
    print('lang model loaded well')
    with torch.no_grad():
        task_emb = lang_model.encode(task_chosen)
    print(f'task emb shape : {task_emb.shape}')
    policy = hydra.utils.instantiate(saved_yaml.agent)
    print()
    args.img_size = saved_yaml.suite.img_size

    ckpt_dir = args.ckpt_dir
    print(f"Loading checkpoint: {ckpt_dir}")
    bc_snapshot = Path(ckpt_dir)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    with bc_snapshot.open("rb") as f:
        payload = torch.load(f,weights_only=False)

    policy.load_snapshot(payload)
    print("Done")

    prompt = task_emb
    max_publish_step = 3000

    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
    debug_pos = [0.0246, 1.2796, -0.8016,  0.0911,  0.9820, -0.1517,  0.0291,  
                 0.0246, 1.2796, -0.8016,  0.0911,  0.9820, -0.1517,  0.0291]

    debug_pos = [ 0.0398,  0.0000, -0.0235, -0.0230,  0.2244,  0.0000,  0.0272,  0.4414,
          1.1391, -0.8630,  0.1986,  0.8499, -0.9494,  0.0450]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    action = None
    aggregator.reset()

    use_nta = args.use_nonlinear_temporal_agg

    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            t = 0
            max_t = 0
            rate = rospy.Rate(args.publish_rate)
            
            while t < max_publish_step and not rospy.is_shutdown():
                if t >= max_t:
                    pre_action = action
                    if pre_action is not None:
                        print(f'pre action shape:{pre_action.shape}')
                    inference_thread = threading.Thread(target=inference_process,
                                                        args=(args, prompt, ros_operator,
                                                                policy, t, pre_action,stats_dict))
                    inference_thread.start()
                    inference_thread.join()
                    inference_lock.acquire()
                    if inference_actions is not None:
                        inference_thread = None
                        action = inference_actions
                        speed = inference_speed
                        inference_action_buffer = inference_actions
                        inference_actions = None
                        inference_speed = None
                        max_t = t + args.pos_lookahead_step
                    inference_lock.release()

                if use_nta:
                    thisaction = (
                            aggregator.record_and_get_current_actions(
                                torch.tensor(action), speed, t
                            )
                        )
                    left_action = thisaction[...,:7]
                    right_action = thisaction[...,7:14]
                    ros_operator.puppet_arm_publish(left_action, right_action)
                    
                elif args.debug_full_traj:
                    print("Done")
                    for i in range(2,20):
                        thisaction = action[i]
                        left_action = thisaction[...,:7]
                        print(f'current left arm {thisaction[...,:7] }')
                        right_action = thisaction[...,7:14]
                        ros_operator.puppet_arm_publish(left_action, right_action)
                        a = input(f"i :")
                else:
                    left_action = action[...,:7]
                    right_action = action[...,7:14]

                    left_action,right_action = action_safety(left_action,right_action)
                    
                    if args.send_action:
                        ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread

                t += 1
                rate.sleep()
                if args.test_debug:
                    a = input("Debug line 427 Enter any key to continue :")
                    if a == 'q':
                        ros_operator.puppet_arm_publish(debug_pos[0:7], debug_pos[7:14])
                        a = input("Debug line 427 Enter any key to continue :")

class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            #print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        situation = 'allpop'  # frame_pop
        if situation != 'allpop':
            while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_deque.popleft()
            while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_deque.popleft()
            while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_deque.popleft()
            while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
                self.puppet_arm_left_deque.popleft()
            while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
                self.puppet_arm_right_deque.popleft()
        else:
            while len(self.img_left_deque) > 1:
                self.img_left_deque.popleft()
            while len(self.img_right_deque) > 1:
                self.img_right_deque.popleft()
            while len(self.img_front_deque) > 1:
                self.img_front_deque.popleft()
            while len(self.puppet_arm_left_deque) > 1:
                self.puppet_arm_left_deque.popleft()
            while len(self.puppet_arm_right_deque) > 1:
                self.puppet_arm_right_deque.popleft()


        img_left = self.img_left_deque.popleft()
        np_arr = np.frombuffer(img_left.data, np.uint8)
        img_left = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_left = img_left[:, :, ::-1]
        img_right = self.img_right_deque.popleft()
        np_arr = np.frombuffer(img_right.data, np.uint8)
        img_right = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_right = img_right[:, :, ::-1]
        img_front = self.img_front_deque.popleft()
        np_arr = np.frombuffer(img_front.data, np.uint8)
        img_front = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_front = img_front[:, :, ::-1]

        puppet_arm_left = self.puppet_arm_left_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()


        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, CompressedImage, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, CompressedImage, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, CompressedImage, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)

@hydra.main(config_path="../cfgs", config_name="inference")
def main(cfg):
    ros_operator = RosOperator(cfg)
    model_inference(cfg, ros_operator, save_episode=True)

if __name__ == '__main__':
    main()
