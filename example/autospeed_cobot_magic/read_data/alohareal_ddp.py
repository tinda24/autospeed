import random
import numpy as np
from pathlib import Path
import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
import time
import torch.distributed as dist
import math
from tqdm import tqdm
import os
import re


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_queries,
        img_size,
        new_loss_args,
        group_loss_window,
        action_overcollect_ratio=5,
        stats_path=None,
        rank=0,
        world_size=1,
        debug=False,
        proprio_dim=None,
    ):
        self.debug = debug
        self.path = path
        self.tasks = tasks[0]["desktop"]
        self.num_queries = num_queries
        self.img_size = img_size
        self.proprio_dim = proprio_dim
        self.stats_path = stats_path
        self.rank = rank
        self.world_size = world_size

        speed_range = new_loss_args.get("speed_range")
        self.a_min, self.a_max = speed_range[0], speed_range[1]
        self.action_overcollect_ratio = action_overcollect_ratio
        self.group_loss_window = group_loss_window

        task_dirs = [os.path.join(path, task) for task in self.tasks]
        all_items = []
        for task_dir in task_dirs:
            if not os.path.exists(task_dir):
                raise FileNotFoundError(f"Task directory {task_dir} not found")
            all_hdf5_files = [str(p) for p in Path(task_dir).rglob("*.hdf5")]
            all_items.extend([f for f in all_hdf5_files if not f.endswith("stats.hdf5")])
        if rank == 0:
            print(f"Found {len(all_items)} hdf5 files")

        self.items = [item for i, item in enumerate(all_items) if (i % world_size == rank % world_size)]
        self.len_episodes = len(self.items)
        print(f"Rank {rank} found {self.len_episodes} hdf5 files")

        self.stats = {"min": 0, "max": 1}
        self.compute_stats(all_items, stats_path, rank)

        self.aug = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        self.resize_aug = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(self.img_size), transforms.ToTensor()]
        )
        self._episodes = []
        self._correspond_task_name = []
        self._correspond_episode_id = []

        if rank == 0:
            print("Initializing dataset with immediate preload...")
        self.preload_episode()

    def compute_stats(self, all_items, stats_path, rank):
        stats_path = Path(stats_path)
        if stats_path.exists():
            if rank == 0:
                print(f"Loading pre-computed stats from {stats_path}")
            with h5py.File(stats_path, "r") as f:
                self.stats["min"] = f["min"][:]
                self.stats["max"] = f["max"][:]
        else:
            if rank == 0:
                actions = []
                for item in all_items:
                    with h5py.File(item, "r") as data:
                        action = data["action"][()]
                    actions.append(action)

                actions = np.concatenate(actions, axis=0)
                self.stats["min"] = np.min(actions, axis=0)
                self.stats["max"] = np.max(actions, axis=0)

                stats_dir = stats_path.parent
                stats_dir.mkdir(parents=True, exist_ok=True)
                tmp_stats_path = stats_path.with_name(stats_path.name + ".tmp")
                with h5py.File(tmp_stats_path, "w") as f:
                    f.create_dataset("min", data=self.stats["min"])
                    f.create_dataset("max", data=self.stats["max"])
                os.replace(tmp_stats_path, stats_path)
                print(f"Stats computed and saved to {stats_path}")
                print(f'  min: {self.stats["min"]}')
                print(f'  max: {self.stats["max"]}')

            if rank != 0:
                print(f"Rank {rank} waiting for stats file...")
                while not stats_path.exists():
                    time.sleep(0.1)
                with h5py.File(stats_path, "r") as f:
                    self.stats["min"] = f["min"][:]
                    self.stats["max"] = f["max"][:]

    def preprocess(self, x):
        return (x - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-5)

    def _preload_single_episode(self, item):
        with h5py.File(item, "r") as data:
            episode = {
                "fix_camera": data["cam_global"][:],
                "left_camera": data["cam_left_wrist"][:],
                "right_camera": data["cam_right_wrist"][:],
                "proprioceptive": data["qpos"][:],
                "actions": data["action"][:],
                "task_emb": data["task_emb"][:],
            }

        return episode

    def preload_episode(self):
        for item in tqdm(self.items, desc="Preloading episodes"):
            task_name = os.path.basename(os.path.dirname(item))
            match = re.search(r"episode_(\d+)\.hdf5$", item)
            episode_id = int(match.group(1)) if match else None

            episode = self._preload_single_episode(item)

            self._episodes.append(episode)
            self._correspond_task_name.append(task_name)
            self._correspond_episode_id.append(episode_id)

            if self.debug:
                print("Debug dataloader: loading one data file only")
                break

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def sample_episode(self):
        sampled_id = random.randint(0, len(self._episodes) - 1)
        sampled_episode = self._episodes[sampled_id]
        sampled_task_name = self._correspond_task_name[sampled_id]
        sampled_episode_id = self._correspond_episode_id[sampled_id]
        return sampled_episode, sampled_task_name, sampled_episode_id

    def _sample(self):
        sampled_episode, sampled_task_name, sampled_episode_id = self.sample_episode()
        num_frames = len(sampled_episode["fix_camera"])
        sample_idx = random.randint(0, num_frames - self.group_loss_window)

        sampled_fix_camera = sampled_episode["fix_camera"][sample_idx : sample_idx + self.group_loss_window]
        sampled_left_camera = sampled_episode["left_camera"][sample_idx : sample_idx + self.group_loss_window]
        sampled_right_camera = sampled_episode["right_camera"][sample_idx : sample_idx + self.group_loss_window]

        if sampled_left_camera.shape[-2] != self.img_size[0] or sampled_left_camera.shape[-1] != self.img_size[1]:
            aug = self.resize_aug
        else:
            aug = self.aug

        sampled_fix_camera = torch.stack([aug(fix_camera) for fix_camera in sampled_fix_camera])
        sampled_left_camera = torch.stack([aug(left_camera) for left_camera in sampled_left_camera])
        sampled_right_camera = torch.stack([aug(right_camera) for right_camera in sampled_right_camera])

        sampled_proprioceptive_state = sampled_episode["proprioceptive"][
            sample_idx : sample_idx + self.group_loss_window
        ]
        sampled_proprioceptive_state = self.preprocess(sampled_proprioceptive_state)

        actions = torch.from_numpy(sampled_episode["actions"]).float()
        last_action = actions[-1:]
        extended_actions = torch.cat([actions, last_action.repeat(500, 1)], dim=0)

        start_idx = self.action_overcollect_ratio * sample_idx
        chunks_needed = math.ceil(self.num_queries * self.a_max * self.action_overcollect_ratio)
        end_idx = start_idx + chunks_needed

        sampled_actions = [
            extended_actions[
                start_idx + j * self.action_overcollect_ratio : end_idx + j * self.action_overcollect_ratio
            ]
            for j in range(self.group_loss_window)
        ]
        sampled_actions = torch.stack(sampled_actions)
        sampled_actions = self.preprocess(sampled_actions)

        return {
            "cam_global": sampled_fix_camera,
            "cam_left_wrist": sampled_left_camera,
            "cam_right_wrist": sampled_right_camera,
            "proprioceptive": sampled_proprioceptive_state,
            "action_future": sampled_actions,
            "task_emb": sampled_episode["task_emb"][()],
        }, sampled_task_name, sampled_episode_id, sample_idx

    def get_norm_stats(self):
        return self.stats

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self.num_frames
