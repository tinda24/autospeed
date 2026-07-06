import random
import numpy as np
import pickle as pkl
from pathlib import Path
import math
import time
import shutil

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

import h5py
import os
import logging

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
if "ROBOSUITE_MODEL_PATH" not in os.environ:
    try:
        import robosuite
        os.environ["ROBOSUITE_MODEL_PATH"] = str(Path(robosuite.__file__).resolve().parent / "models")
    except ImportError:
        pass

def _compute_actions_minmax(pkl_paths):
    mins, maxs = None, None
    for p in pkl_paths:
        data = pkl.load(open(str(p), "rb"))
        for acts in data["actions"]:
            acts = np.asarray(acts, dtype=np.float32)
            if acts.ndim == 1:
                acts = acts[:, None]
            cur_min, cur_max = acts.min(axis=0), acts.max(axis=0)
            mins = cur_min if mins is None else np.minimum(mins, cur_min)
            maxs = cur_max if maxs is None else np.maximum(maxs, cur_max)
    if mins is None:
        raise RuntimeError("No actions found in pkl files")
    return mins.astype(np.float32), maxs.astype(np.float32)


def _load_stats(path):
    with h5py.File(path, "r") as f:
        return f["min"][:].astype(np.float32), f["max"][:].astype(np.float32)


def _save_stats(path, mins, maxs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp"
    tmp.unlink(missing_ok=True)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("min", data=mins)
        f.create_dataset("max", data=maxs)
    os.replace(tmp, path)


def _ensure_action_stats(pkl_paths, dataset_path, run_path, rank):
    if run_path.exists():
        return _load_stats(run_path)

    if rank == 0:
        if dataset_path.exists():
            run_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = run_path.parent / f"{run_path.name}.tmp"
            tmp.unlink(missing_ok=True)
            shutil.copyfile(dataset_path, tmp)
            os.replace(tmp, run_path)
        else:
            mins, maxs = _compute_actions_minmax(pkl_paths)
            _save_stats(dataset_path, mins, maxs)
            _save_stats(run_path, mins, maxs)

    while not run_path.exists():
        time.sleep(0.1)
    return _load_stats(run_path)


def _crop14(img):
    _, h, w = img.shape
    nh, nw = (h // 14) * 14, (w // 14) * 14
    if nh == h and nw == w:
        return img
    top, left = (h - nh) // 2, (w - nw) // 2
    return transforms.functional.crop(img, top, left, nh, nw)

class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite=None,
        scenes=None,                               
        tasks=None,                                                               
        num_demos_per_task=None,
        num_future_queries=1,
        num_past_queries=0,
        img_size=None,
        proprio_dim=None,
        work_dir=None,
        rank=0,
        world_size=1,
        new_loss_args=None,
        action_overcollect_ratio=1,
        stats_path=None,

        group_loss_window=5,
    ):
        self.rank = rank
        self.world_size = world_size
        self.stats_path = stats_path
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        self.num_future_queries = num_future_queries
        self.num_past_queries = num_past_queries
        self.work_dir = work_dir
        self.img_size = img_size if img_size is not None else [128, 128]
        self.proprio_dim = proprio_dim

        new_loss_args = new_loss_args or {"speed_range": [1.0, 1.0]}
        speed_range = new_loss_args.get("speed_range")
        self.a_min, self.a_max = speed_range[0], speed_range[1]

        self.action_overcollect_ratio = action_overcollect_ratio

        self.group_loss_window = group_loss_window

        self._future_window = (
            int(math.ceil(self.num_future_queries * self.a_max * self.action_overcollect_ratio))
            if self.num_future_queries > 0
            else 0
        )
        self._past_window = (
            int(math.ceil(self.num_past_queries * self.a_max * self.action_overcollect_ratio))
            if self.num_past_queries > 0
            else 0
        )
        self.img_transform = self._preprocess_image

        task_lookup = {}
        if tasks:
            print('Load tasks: ', tasks)
            for scene in tasks:
                for scene_name, scene_tasks in scene.items():
                    if scene_tasks is None:
                        continue
                    task_lookup[scene_name] = scene_tasks

        task_names = []                         
        if scenes:
            print('Load scenes: ', scenes)
            for scene_name in scenes:
                scene_tasks = task_lookup.get(scene_name)
                if not scene_tasks:
                    continue
                task_names.extend(scene_tasks)

        data_root = Path(path)
        if suite:
            data_root = data_root / suite
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset directory {data_root} not found")

        discovered_paths = sorted(data_root.glob("*.pkl"))
        if not discovered_paths:
            raise FileNotFoundError(f"No .pkl files found under {data_root}")

        if task_names:                     
            self._paths = {}
            for p in discovered_paths:
                task = p.stem
                if task in task_names:
                    idx = task_names.index(task)
                    self._paths[idx] = p
        else:                                  
            self._paths = {idx: p for idx, p in enumerate(discovered_paths)}

        if not self._paths:
            raise RuntimeError(
                "No task files were matched. Please check scenes/tasks configuration."
            )

        dataset_stats_path = data_root / "stats.hdf5"
        run_stats_path = Path(self.stats_path) if self.stats_path is not None else (Path.cwd() / "stats.hdf5")
        mins, maxs = _ensure_action_stats(discovered_paths, dataset_stats_path, run_stats_path, self.rank)

        if mins.shape[-1] != maxs.shape[-1]:
            raise ValueError(
                f"stats.hdf5 min/max dimension mismatch: {mins.shape} vs {maxs.shape}"
            )

        self.action_mins = mins
        self.action_maxs = maxs

        self.stats = {
            "actions": {"min": -1.0, "max": 1.0},
            "proprioceptive": {"min": 0.0, "max": 1.0},
        }

        self._episodes = {}                              
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:                       
            path_obj = self._paths[_path_idx]
            print(f"Loading {path_obj}")
            data = pkl.load(open(str(path_obj), "rb"))
            observations = data["observations"]
            actions = data["actions"]
            task_emb = torch.as_tensor(data["task_emb"], dtype=torch.float32)

            episode_len = len(observations)
            max_demos = episode_len if num_demos_per_task is None else min(
                num_demos_per_task, episode_len
            )

            print(f"Number of demos loaded: {max_demos}")

            self._episodes[_path_idx] = []
            for i in range(max_demos):                                           
                observation_i = observations[i]
                action_i_raw = np.asarray(actions[i], dtype=np.float32)
                if action_i_raw.ndim != 2:
                    raise ValueError(
                        f"Expected action with shape [T, D], got {action_i_raw.shape} "
                        f"in {path_obj} demo {i}"
                    )
                if action_i_raw.shape[-1] != self.action_mins.shape[-1]:
                    raise ValueError(
                        f"Action dim {action_i_raw.shape[-1]} does not match stats dim "
                        f"{self.action_mins.shape[-1]} in {path_obj} demo {i}"
                    )
                action_i = self._pad_actions(action_i_raw)
                episode = dict(
                    observation=observation_i,
                    action=action_i,
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                episode_len = len(observation_i["pixels"])
                self._max_episode_len = max(self._max_episode_len, episode_len)
                joint_dim = observation_i["joint_states"].shape[-1]
                gripper_dim = observation_i["gripper_states"].shape[-1]
                self._max_state_dim = max(
                    self._max_state_dim, joint_dim + gripper_dim
                )
                self._num_samples += episode_len

        self.envs_till_idx = len(self._episodes)                  


    def _normalize_actions(self, x):
        x = np.asarray(x, dtype=np.float32)
        ranges = np.maximum(self.action_maxs - self.action_mins, 1e-6)
        return np.clip((x - self.action_mins) / ranges * 2.0 - 1.0, -1.0, 1.0)

    def _preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            x = img.permute(2, 0, 1) if img.ndim == 3 and img.shape[-1] in (1, 3, 4) else img
            x = x.unsqueeze(0) if x.ndim == 2 else x
            x = x.float() / 255.0 if x.max() > 1.0 else x.float()
        else:
            x = transforms.functional.to_tensor(img)
        x = transforms.functional.resize(x, self.img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        return _crop14(x)

    def _pad_actions(self, actions):
        if actions.ndim == 1:
            actions = actions[:, None]
        front = np.repeat(actions[:1], self._past_window, axis=0) if self._past_window else np.empty((0, actions.shape[-1]))
        back = np.repeat(actions[-1:], self._future_window, axis=0) if self._future_window else np.empty((0, actions.shape[-1]))
        return np.concatenate([front, actions, back], axis=0)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx
        task = self._episodes[idx]

        demo_id = random.randint(0, len(task) - 1)
        episode = task[demo_id]
                                      
        return (episode, idx, demo_id) if env_idx is None else episode

    def _sample(self):
                                                              
        episode, env_idx, demo_id = self._sample_episode()
        observations = episode["observation"]
        actions = episode["action"]
        task_emb = episode["task_emb"]

        num_frames = len(observations["pixels"])
        sample_idx = np.random.randint(0, num_frames - self.group_loss_window)

        sampled_pixel = observations["pixels"][sample_idx:sample_idx+self.group_loss_window]
        sampled_pixel_egocentric = observations["pixels_egocentric"][sample_idx:sample_idx+self.group_loss_window]

        sampled_proprioceptive_state = np.concatenate(
            [
                observations["joint_states"][sample_idx:sample_idx+self.group_loss_window],
                observations["gripper_states"][sample_idx:sample_idx+self.group_loss_window]
            ],
            axis=-1,
        )

        action_center_idx = self._past_window + sample_idx * self.action_overcollect_ratio
        past_start = action_center_idx - self._past_window
        future_end = action_center_idx + self._future_window
        action_future_list = []
        for j in range(self.group_loss_window):
            offset = j * self.action_overcollect_ratio
            action_future_list.append(actions[action_center_idx + offset : future_end + offset])

        sample = {
            "high_camera": torch.stack([self.img_transform(pixel) for pixel in sampled_pixel]),
            "right_camera": torch.stack([self.img_transform(pixel) for pixel in sampled_pixel_egocentric]),
            "proprioceptive": torch.from_numpy(sampled_proprioceptive_state).float(),
            "action_future": torch.from_numpy(
                self._normalize_actions(np.stack(action_future_list, axis=0))
            ).float(),
            "task_emb": task_emb.clone(),
        }

        if self.num_past_queries > 0:
            action_past = actions[past_start:action_center_idx]
            sample["action_past"] = torch.from_numpy(self._normalize_actions(action_past)).float()

        self.logger.debug(f"action_center_idx: {action_center_idx}")
        self.logger.debug(f"past_start: {past_start}")
        self.logger.debug(f"future_end: {future_end}")

                       
        return sample, env_idx, demo_id, sample_idx

    def get_specific_episode(self, env_idx, episode_idx):
        picked_episode = self._episodes[env_idx][episode_idx]

        path_dir = self._paths[env_idx]
        print(f"path_dir: {path_dir}")
        task_name = path_dir.stem

        observations = picked_episode["observation"]
        task_emb = picked_episode["task_emb"]

        sampled_pixel = torch.stack(
            [self.img_transform(pixel) for pixel in observations["pixels"]]
        )
        sampled_pixel_egocentric = torch.stack(
            [self.img_transform(pixel) for pixel in observations["pixels_egocentric"]]
        )
        sampled_proprioceptive_state = np.concatenate(
            [
                observations["joint_states"],
                observations["gripper_states"]
            ],
            axis=-1,
        )

        assumed_episode = {
            "pixels": sampled_pixel,
            "pixels_egocentric": sampled_pixel_egocentric,
            "proprioceptive": sampled_proprioceptive_state,
        }

        return assumed_episode, task_emb, task_name

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples

    def get_norm_stats(self):
        return self.stats
