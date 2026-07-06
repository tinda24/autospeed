import numpy as np
import torch
import os
import cv2
import h5py
import matplotlib as mpl
from torch.utils.data import TensorDataset, DataLoader


def relabel_waypoints(arr, waypoint_indices):
    start_idx = 0
    for key_idx in waypoint_indices:
                                                                                       
        arr[start_idx:key_idx] = arr[key_idx]
        start_idx = key_idx
    return arr

def put_text(img, text, is_waypoint=False, font_size=1, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
                                                 
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img

def bottom_20_percent_value(lst):

    lst = list(lst)
    sorted_lst = sorted(lst)

    bottom_20_index = int(len(sorted_lst)*0.2)

    return sorted_lst[bottom_20_index]

def plot_3d_trajectory(ax, traj_list, actions_var_norm=None, distance=None, label=None, gripper=None, legend=True, add=None):
    mark = None
    if actions_var_norm is not None:
        import math
        actions_var_log = actions_var_norm
        mark = np.array(actions_var_log)
        key = bottom_20_percent_value(mark[:220])
    elif distance is not None:
        mark = [d*50 for d in distance]

    l = label
    num_frames = len(traj_list)
    count = 0
    for i in range(num_frames):
                                                       
        gripper_state_changed = (
            gripper is not None and i > 0 and gripper[i] != gripper[i - 1]
        )
        if label == "pred" or label == "waypoints":
            if mark is None:
                if gripper_state_changed or (add is not None and i in add):
                    c = mpl.cm.Oranges(0.2 + 0.5 * i / num_frames)
                else:
                                                                 
                    c = mpl.cm.Reds(0.1)
            else:
                c = mpl.cm.Reds(np.clip((0.5-mark[i]),0,1))
        elif label == "gt" or label == "ground truth" or label == "demos replay":
            if mark is None:
                if gripper_state_changed:
                    c = mpl.cm.Greens(0.2 + 0.5 * i / num_frames)
                else:
                    c = mpl.cm.Blues(0.9 + 0.1 * i / num_frames)
            else:
                c = mpl.cm.Blues(0.5+0.5*np.clip((0.5-mark[i]),0,1))
        else:
                                                               
                if mark[i] < 0.82 and i>20 and i<240:                                       
                    c = mpl.cm.Blues(np.clip((1-mark[i]),0,1))
                    count+=1
                else:
                    c = mpl.cm.Reds(np.clip((1-mark[i]),0,1))

                                                        
        if gripper_state_changed:
            if gripper[i] == 1:        
                marker = "D"
            else:         
                marker = "s"
        else:
            marker = "o"

                                                                    
        if (label == "pred" or label == "action" or label == "waypoints") and i > 0:
            v = traj_list[i] - traj_list[i - 1]
            ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color="r",
                alpha=0.5,
                              
            )

                                                                      
        if add is not None and i in add:
            marker = "D"
            ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                markersize=10,
            )
        else:
            if i > 0:
                v = traj_list[i] - traj_list[i - 1]
                ax.quiver(
                traj_list[i - 1][0],
                traj_list[i - 1][1],
                traj_list[i - 1][2],
                v[0],
                v[1],
                v[2],
                color=c,
                alpha=0.5,
                              
            )
            if gripper_state_changed:
                if gripper[i] == 1:        
                   marker = "D"
                else:         
                   marker = "s"
                ax.plot(
                [traj_list[i][0]],
                [traj_list[i][1]],
                [traj_list[i][2]],
                marker=marker,
                label=l,
                color=c,
                markersize=5,
                )
        l = None

    if legend:
        ax.legend()


def process_action_label(action, label, is_pad):
    low_v = 2
    high_v = 4
    horizon, dim = action.shape
    new_actions = torch.zeros_like(action)
    new_labels = torch.zeros_like(label)
    new_is_pad = torch.zeros_like(is_pad)

    current_action = action                         
    current_label = label                     
    current_is_pad = is_pad

    indices = []
    i = -1
    while i < horizon:
        if current_label[i] == 0 and i+low_v < horizon:
            i += low_v                     
            indices.append(i)
        elif current_label[i] == 1:
                                                          
            if i + high_v < horizon and torch.all(current_label[i:i + high_v] == 1):
                i += high_v                            
                indices.append(i)
            else:
                                                      
                next_zero = (current_label[i + 1:] == 0).nonzero(as_tuple=True)[0]
                if len(next_zero) > 0:
                    i = i + 1 + next_zero[0].item()
                    indices.append(i)
                else:
                    break                    
        else:
            i += 1

                                                     
    new_actions[:len(indices)] = current_action[indices]
    new_labels[:len(indices)] = current_label[indices]
    new_is_pad[:len(indices)] = current_is_pad[indices]

    return new_actions, new_is_pad

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        speedup=False,
        constant_waypoint=None,
        policy_class = "ACT",
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.speedup = speedup
        self.constant_waypoint = constant_waypoint
        self.policy_class = policy_class
        self.__getitem__(0)                          

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False            

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
                                              
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
                                                          
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]                                        
                action_len = episode_len - max(
                    0, start_ts - 1
                )                                        


        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

                                        
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

                                
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

                      
        image_data = torch.einsum("k h w c -> k c h w", image_data)

                                                   
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]
                                          
        if self.speedup:
            with h5py.File(dataset_path, "r") as root:
                if self.policy_class == "ACT":
                    original_label_shape = root["/labels"].shape
                    label = root["/labels"][start_ts:]
                elif self.policy_class == "DP":
                    original_label_shape = root["/labels_dp"].shape
                    label = root["/labels_dp"][start_ts:]
            padded_labels = np.zeros(original_label_shape, dtype=np.float32)
            padded_labels[:action_len] = label
            label_data = torch.from_numpy(padded_labels).float()
            action_data, is_pad = process_action_label(action_data, label_data, is_pad)
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            qvel = root["/observations/qvel"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

                           
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10)            
    action_max, _ = torch.max(all_action_data, 0, keepdim=True)
    action_max,_ = torch.max(action_max,1, keepdim=True)
    action_min, _ = torch.min(all_action_data, 0, keepdim=True)
    action_min,_ = torch.min(action_min,1,keepdim=True)
                  
    scale = (action_max - action_min)/2
    offset = action_min - (action_max-action_min)/2

                         
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10)            
    qpos_max = torch.max(torch.abs(all_qpos_data))
    qpos_min = torch.zeros_like(qpos_max)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": 2*action_std.numpy().squeeze(),                                         
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }
    return stats


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    speedup=False,
    constant_waypoint=None,
    policy_class = "ACT",
    num_workers=1,
):
    print(f"\nData from: {dataset_dir}\n")
                             
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

                                                    
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

                                      
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        speedup = speedup,
        constant_waypoint=constant_waypoint,
        policy_class = policy_class
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        speedup = speedup,
        constant_waypoint=constant_waypoint,
        policy_class = policy_class
    )

    train_loader_kwargs = {}
    val_loader_kwargs = {}
    if num_workers and int(num_workers) > 0:
        train_loader_kwargs["prefetch_factor"] = 1
        val_loader_kwargs["prefetch_factor"] = 1

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        **train_loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        **val_loader_kwargs,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


             


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
         
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

            
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


                    


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
