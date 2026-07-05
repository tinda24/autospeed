import h5py
import torch
import torchvision.transforms as transforms
import os
import re
import math
from tqdm import tqdm


class AloharealEvalEpisodeLoader:
    def __init__(
        self,
        eval_data_paths,
        stats_path=None,
        img_size=(224, 224),
        num_queries=16,
        speed_range=(0.5, 2.0),
        action_overcollect_ratio=2,
    ):
        self.eval_data_paths = eval_data_paths
        self.stats_path = stats_path
        self.img_size = img_size
        self.num_queries = num_queries
        self.speed_range = speed_range
        self.action_overcollect_ratio = action_overcollect_ratio

        assert os.path.exists(stats_path)
        with h5py.File(stats_path, "r") as f:
            self.stats = {
                "min": f["min"][:],
                "max": f["max"][:],
            }

        self.aug = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        self.resize_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )

        self.episodes = []

        print("Loading eval episodes...")
        for eval_data_path in tqdm(self.eval_data_paths):
            self.episodes.append(self.preload_episode(eval_data_path))

    def preload_episode(self, eval_data_path):
        with h5py.File(eval_data_path, "r") as f:
            actions = f["action"][:]
            fix_camera = f["cam_global"][:]
            left_camera = f["cam_left_wrist"][:]
            right_camera = f["cam_right_wrist"][:]
            proprioceptive = f["qpos"][:]
            task_emb = f["task_emb"][:]

            num_frames = len(proprioceptive)

            if left_camera.shape[-2] != self.img_size[0] or left_camera.shape[-1] != self.img_size[1]:
                aug_func = self.resize_aug
            else:
                aug_func = self.aug

            fix_camera_tensor = torch.stack([aug_func(img) for img in fix_camera])
            left_camera_tensor = torch.stack([aug_func(img) for img in left_camera])
            right_camera_tensor = torch.stack([aug_func(img) for img in right_camera])

            stats_min = torch.from_numpy(self.stats["min"])
            stats_max = torch.from_numpy(self.stats["max"])
            proprioceptive_tensor = torch.from_numpy(proprioceptive).float()
            proprioceptive_tensor = (proprioceptive_tensor - stats_min) / (stats_max - stats_min + 1e-5)

            _, a_max = self.speed_range
            actions_raw = torch.from_numpy(actions).float()
            last_action = actions_raw[-1:]
            extended_actions = torch.cat([actions_raw, last_action.repeat(500, 1)], dim=0)
            max_chunk_length = math.ceil(self.num_queries * a_max * self.action_overcollect_ratio)

            action_chunks = []
            for j in tqdm(range(num_frames)):
                start_idx = self.action_overcollect_ratio * j
                end_idx = start_idx + max_chunk_length
                action_chunks.append(extended_actions[start_idx:end_idx])

            action_chunks_tensor = torch.stack(action_chunks, dim=0)
            action_chunks_tensor = (action_chunks_tensor - stats_min) / (stats_max - stats_min + 1e-5)

            task_name = os.path.basename(os.path.dirname(eval_data_path))
            match = re.search(r"episode_(\d+)\.hdf5$", eval_data_path)
            episode_id = int(match.group(1)) if match else None

            obs = {
                "cam_global": fix_camera_tensor,
                "cam_left_wrist": left_camera_tensor,
                "cam_right_wrist": right_camera_tensor,
                "proprioceptive": proprioceptive_tensor,
            }

            return {
                "num_frames": num_frames,
                "task_emb": task_emb,
                "obs": obs,
                "action_chunks": action_chunks_tensor,
                "task_name": task_name,
                "episode_id": episode_id,
            }

    def sample_from_filename(self, eval_data_path):
        index = self.eval_data_paths.index(eval_data_path)
        return self.episodes[index]
