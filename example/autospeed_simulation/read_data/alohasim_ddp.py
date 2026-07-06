from torch.utils.data import IterableDataset
import numpy as np
from pathlib import Path
import h5py
import random
import torch
import torchvision.transforms as transforms
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
        action_overcollect_ratio = 5,
        stats_path=None,
        rank=0,
        world_size=1,
        debug = False,
        proprio_dim=None,
     ):
        self.debug = debug
        self.path = path
        self.tasks = tasks

        self.num_queries = num_queries

        self.img_size = img_size
        self.stats_path = stats_path
        self.proprio_dim = proprio_dim

        self.rank = rank
        self.world_size = world_size

        speed_range = new_loss_args.get("speed_range")
        self.a_min, self.a_max = speed_range[0], speed_range[1]

        self.action_overcollect_ratio = action_overcollect_ratio

        self.group_loss_window = group_loss_window

        all_items = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task directory {path} not found")

        task_names = []
        print(f'tasks: {tasks}')
        for scene in tasks:
             print(f'scene: {scene}')
             for v in scene.values():
                  task_names.extend([str(t) for t in v])

        all_hdf5_files = []
        for task in task_names:
             task_dir = Path(path) / task
             if task_dir.exists():
                  all_hdf5_files.extend([str(p) for p in task_dir.rglob("*.hdf5")])
             else:
                  if rank == 0:
                       print(f"Warning: task dir not found: {task_dir}")

        all_items.extend([f for f in all_hdf5_files if not f.endswith('stats.hdf5')])
        if rank == 0:
            print(f'found{len(all_items)} hdf5files')

        self.items = [item for i, item in enumerate(all_items) if (i % world_size == rank % world_size)]
        self.len_episodes = len(self.items)
        print(f'rank {rank} found{self.len_episodes} hdf5files')

        self.stats = {"min": 0, "max": 1}
        self.compute_stats(all_items, stats_path, rank)

        self.aug = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),])
        self.resize_aug = transforms.Compose([transforms.ToPILImage(),transforms.Resize(self.img_size),transforms.ToTensor(),])
        self._episodes = []
        self._correspond_task_name = []
        self._correspond_episode_id = []

        if rank == 0:
            print(f'Initializing dataset with immediate preload...')
        self.preload_episode()


    def compute_stats(self, all_items, stats_path, rank):
        stats_path = Path(stats_path)
        if stats_path.exists():
            if rank == 0:
                print(f'Loading pre-computed stats from {stats_path}')
            with h5py.File(stats_path, 'r') as f:
                self.stats["min"] = f['min'][:]
                self.stats["max"] = f['max'][:]
        else:
            if rank == 0:

                actions = []
                for i, item in enumerate(all_items):
                    with h5py.File(item, "r") as data:
                        action = data["action"][()]
                    actions.append(action)

                actions = np.concatenate(actions, axis=0)
                self.stats["min"] = np.min(actions, axis=0)
                self.stats["max"] = np.max(actions, axis=0)

                stats_dir = stats_path.parent
                stats_dir.mkdir(parents=True, exist_ok=True)
                tmp_stats_path = stats_path.with_name(stats_path.name + ".tmp")
                with h5py.File(tmp_stats_path, 'w') as f:
                    f.create_dataset('min', data=self.stats["min"])
                    f.create_dataset('max', data=self.stats["max"])
                os.replace(tmp_stats_path, stats_path)
                print(f'Stats computed and saved to {stats_path}')
                print(f'  min: {self.stats["min"]}')
                print(f'  max: {self.stats["max"]}')

            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier(device_ids=[self.rank])

            if rank != 0:
                print(f'Rank {rank} waiting for stats file... {stats_path}')
                while not stats_path.exists():
                    time.sleep(0.1)
                with h5py.File(stats_path, 'r') as f:
                    self.stats["min"] = f['min'][:]
                    self.stats["max"] = f['max'][:]

    def preprocess(self, x):
                                                                                         
        return 2 * (x - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + 1e-5) - 1

    def _preload_single_episode(self, item):
        with h5py.File(item, "r") as data:
            actions = data["action"][:]

            fix_camera = data["observations/images/top"][:]

            proprioceptive = data["observations/qpos"][:]

            episode = {
                "fix_camera": fix_camera,
                "proprioceptive": proprioceptive,
                "actions": actions,
                "task_emb": data["task_emb"][:],
            }

        return episode

    def preload_episode(self):
        for item in tqdm(self.items, desc='Preloading episodes'):
            task_name = os.path.basename(os.path.dirname(item))
            match = re.search(r"episode_(\d+)\.hdf5$", item)
            episode_id = int(match.group(1)) if match else None

            episode = self._preload_single_episode(item)

            self._episodes.append(episode)
            self._correspond_task_name.append(task_name)
            self._correspond_episode_id.append(episode_id)

            if self.debug:
                print(f'Debug Datalaoder, only load ONE data file')
                print(f'Debug Datalaoder, only load ONE data file')
                print(f'Debug Datalaoder, only load ONE data file')
                print(f'Debug Datalaoder, only load ONE data file')
                print(f'Debug Datalaoder, only load ONE data file')
                print(f'Debug Datalaoder, only load ONE data file')
                print(f'Debug Datalaoder, only load ONE data file')
                break                     

        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier(device_ids=[self.rank])

    def sample_episode(self):
        sampled_id = random.randint(0, len(self._episodes) - 1)
        sampled_episode = self._episodes[sampled_id]
        sampled_task_name = self._correspond_task_name[sampled_id]
        sampled_episode_id = self._correspond_episode_id[sampled_id]
        return sampled_episode,sampled_task_name,sampled_episode_id

    def _sample(self):
        sampled_episode, sampled_task_name, sampled_episode_id = self.sample_episode()
        num_frames = len(sampled_episode["fix_camera"])
        sample_idx = random.randint(0, num_frames - self.group_loss_window)

               
        sampled_fix_camera = sampled_episode["fix_camera"][sample_idx:sample_idx+self.group_loss_window]

        if sampled_fix_camera.shape[-2] != self.img_size[0] or sampled_fix_camera.shape[-1] != self.img_size[1]:
            aug = self.resize_aug
        else:
            aug = self.aug

        sampled_fix_camera = torch.stack([aug(fix_camera) for fix_camera in sampled_fix_camera])           

        sampled_proprioceptive_state = sampled_episode["proprioceptive"][sample_idx:sample_idx+self.group_loss_window]
        sampled_proprioceptive_state = self.preprocess(sampled_proprioceptive_state)      
       
        actions = torch.from_numpy(sampled_episode["actions"]).float()
        last_action = actions[-1:]
        extended_actions = torch.cat([actions, last_action.repeat(500, 1)], dim=0)

        start_idx = self.action_overcollect_ratio * sample_idx
        chunks_needed = math.ceil(self.num_queries * self.a_max * self.action_overcollect_ratio)
        end_idx = start_idx + chunks_needed

        sampled_actions = [extended_actions[start_idx+j*self.action_overcollect_ratio:end_idx+j*self.action_overcollect_ratio] for j in range(self.group_loss_window)]
        sampled_actions = torch.stack(sampled_actions)        
        sampled_actions = self.preprocess(sampled_actions)

        return {
            "cam_global": sampled_fix_camera,
            "proprioceptive": sampled_proprioceptive_state,
            "action_future": sampled_actions,
            "task_emb": sampled_episode["task_emb"][()],
        }, sampled_task_name, sampled_episode_id, sample_idx

    def __iter__(self):
        while True:
            yield self._sample()
