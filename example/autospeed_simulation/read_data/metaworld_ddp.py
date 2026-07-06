from __future__ import annotations

import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class BCDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        tasks: List[Dict[str, List[str]]],
        num_queries: int,
        img_size: List[int],
        new_loss_args: Dict,
        group_loss_window: int,
        action_overcollect_ratio: int = 1,
        stats_path: str | Path | None = None,
        rank: int = 0,
        world_size: int = 1,
        debug: bool = False,
        proprio_dim: int | None = 9,
    ) -> None:
        self.debug = debug
        self.path = Path(path)
        self.tasks = tasks
        self.num_queries = num_queries
        self.img_size = img_size
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.proprio_dim = int(proprio_dim or 9)
        self.rank = rank
        self.world_size = world_size

        speed_range = new_loss_args.get("speed_range")
        if speed_range is None or len(speed_range) != 2:
            raise ValueError("new_loss_args.speed_range must be provided with [min, max]")
        self.a_min, self.a_max = float(speed_range[0]), float(speed_range[1])

        self.action_overcollect_ratio = int(action_overcollect_ratio)
        self.group_loss_window = int(group_loss_window)
        self.chunk_len = int(
            math.ceil(self.num_queries * self.a_max * self.action_overcollect_ratio)
        )

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

        task_names = self._flatten_task_names(tasks)
        all_items = self._discover_hdf5_files(task_names)
        if len(all_items) == 0:
            raise FileNotFoundError(
                f"No Metaworld episodes found under {self.path} for tasks={task_names}"
            )

        self.items = [
            item
            for idx, item in enumerate(all_items)
            if idx % self.world_size == self.rank % self.world_size
        ]
        if len(self.items) == 0:
            raise RuntimeError(
                f"Rank {self.rank} has no episodes after sharding. "
                f"total_episodes={len(all_items)}, world_size={self.world_size}"
            )

        if self.rank == 0:
            print(f"[MetaWorldBCDataset] total episodes: {len(all_items)}")
        print(f"[MetaWorldBCDataset] rank {self.rank} episodes: {len(self.items)}")

        if self.stats_path is None:
            raise ValueError("stats_path must be provided")
        (
            self.action_min,
            self.action_max,
            self.proprio_min,
            self.proprio_max,
        ) = self._load_or_compute_stats(all_items, self.stats_path)

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.resize_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        self._episodes: List[Dict[str, np.ndarray]] = []
        self._task_names: List[str] = []
        self._episode_ids: List[int] = []
        self._preload_episodes()

    def _flatten_task_names(self, tasks: List[Dict[str, List[str]]]) -> List[str]:
        task_names: List[str] = []
        for scene in tasks:
            for _, task_list in scene.items():
                task_names.extend([str(task) for task in task_list])
        return task_names

    def _discover_hdf5_files(self, task_names: List[str]) -> List[str]:
        all_files: List[str] = []
        for task in task_names:
            task_dir = self.path / task
            if task_dir.exists():
                search_root = task_dir
            elif self.path.name == task and self.path.exists():
                                                                                  
                search_root = self.path
            else:
                if self.rank == 0:
                    print(
                        f"[MetaWorldBCDataset] warning: task dir not found for task={task}. "
                        f"checked={task_dir} and base_path={self.path}"
                    )
                continue

            files = sorted(str(p) for p in search_root.rglob("*.hdf5"))
            all_files.extend(files)
        return [f for f in all_files if not f.endswith("stats.hdf5")]

    def _load_or_compute_stats(
        self,
        all_items: List[str],
        stats_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if stats_path.exists():
            if self.rank == 0:
                print(f"[MetaWorldBCDataset] loading stats from {stats_path}")
            with h5py.File(stats_path, "r") as f:
                action_min = f["action_min"][:].astype(np.float32)
                action_max = f["action_max"][:].astype(np.float32)
                proprio_min = f["proprio_min"][:].astype(np.float32)
                proprio_max = f["proprio_max"][:].astype(np.float32)
            return action_min, action_max, proprio_min, proprio_max

        if self.rank == 0:
            action_min = None
            action_max = None
            proprio_min = None
            proprio_max = None

            for item in all_items:
                with h5py.File(item, "r") as data:
                    actions = data["action"][:].astype(np.float32)
                    qpos = data["observations/qpos"][:, : self.proprio_dim].astype(np.float32)

                cur_action_min = np.min(actions, axis=0)
                cur_action_max = np.max(actions, axis=0)
                cur_proprio_min = np.min(qpos, axis=0)
                cur_proprio_max = np.max(qpos, axis=0)

                action_min = cur_action_min if action_min is None else np.minimum(action_min, cur_action_min)
                action_max = cur_action_max if action_max is None else np.maximum(action_max, cur_action_max)
                proprio_min = cur_proprio_min if proprio_min is None else np.minimum(proprio_min, cur_proprio_min)
                proprio_max = cur_proprio_max if proprio_max is None else np.maximum(proprio_max, cur_proprio_max)

            if action_min is None or proprio_min is None:
                raise RuntimeError("Failed to compute stats: empty dataset")

            stats_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = stats_path.with_name(f"{stats_path.name}.tmp")
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("action_min", data=action_min)
                f.create_dataset("action_max", data=action_max)
                f.create_dataset("proprio_min", data=proprio_min)
                f.create_dataset("proprio_max", data=proprio_max)
            os.replace(tmp_path, stats_path)
            print(f"[MetaWorldBCDataset] computed stats and saved to {stats_path}")

        self._maybe_barrier()

        if self.rank != 0:
            while not stats_path.exists():
                time.sleep(0.1)

        with h5py.File(stats_path, "r") as f:
            action_min = f["action_min"][:].astype(np.float32)
            action_max = f["action_max"][:].astype(np.float32)
            proprio_min = f["proprio_min"][:].astype(np.float32)
            proprio_max = f["proprio_max"][:].astype(np.float32)

        return action_min, action_max, proprio_min, proprio_max

    def _normalize_action(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.action_min) / (self.action_max - self.action_min + 1e-5) - 1.0

    def _normalize_proprio(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.proprio_min) / (self.proprio_max - self.proprio_min + 1e-5) - 1.0

    def _load_one_episode(self, item: str) -> Dict[str, np.ndarray]:
        with h5py.File(item, "r") as data:
            actions = data["action"][:].astype(np.float32)
            top_camera = data["observations/images/top"][:]
            proprio = data["observations/qpos"][:, : self.proprio_dim].astype(np.float32)
            task_emb = data["task_emb"][:].astype(np.float32)

        if actions.shape[1] != 4:
            raise ValueError(f"Expected action dim 4, got {actions.shape} in {item}")
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio dim {self.proprio_dim}, got {proprio.shape} in {item}"
            )
        if not (len(actions) == len(proprio) == len(top_camera)):
            raise ValueError(
                "Length mismatch in episode "
                f"{item}: len(action)={len(actions)}, len(qpos)={len(proprio)}, len(images)={len(top_camera)}"
            )

        return {
            "actions": actions,
            "top_camera": top_camera,
            "proprio": proprio,
            "task_emb": task_emb,
        }

    def _preload_episodes(self) -> None:
        for item in tqdm(self.items, desc="Preloading Metaworld episodes"):
            task_name = Path(item).parent.name
            match = re.search(r"episode_(\d+)\.hdf5$", item)
            episode_id = int(match.group(1)) if match else -1

            episode = self._load_one_episode(item)
            self._episodes.append(episode)
            self._task_names.append(task_name)
            self._episode_ids.append(episode_id)

            if self.debug:
                print("[MetaWorldBCDataset] debug=True, only preloading one episode")
                break

        self._maybe_barrier()

    def _maybe_barrier(self) -> None:
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _sample_episode(self) -> Tuple[Dict[str, np.ndarray], str, int]:
        sampled_id = random.randint(0, len(self._episodes) - 1)
        return (
            self._episodes[sampled_id],
            self._task_names[sampled_id],
            self._episode_ids[sampled_id],
        )

    def _sample(self):
        episode, task_name, episode_id = self._sample_episode()

        num_frames = len(episode["top_camera"])
        if num_frames < self.group_loss_window:
            raise ValueError(
                f"Episode too short: num_frames={num_frames}, group_loss_window={self.group_loss_window}"
            )

        sample_idx = random.randint(0, num_frames - self.group_loss_window)

        sampled_top = episode["top_camera"][sample_idx : sample_idx + self.group_loss_window]
        h, w = sampled_top.shape[1], sampled_top.shape[2]
        use_resize = h != self.img_size[0] or w != self.img_size[1]
        aug = self.resize_aug if use_resize else self.aug
        sampled_top_tensor = torch.stack([aug(img) for img in sampled_top])                

        sampled_proprio = episode["proprio"][sample_idx : sample_idx + self.group_loss_window]
        sampled_proprio = self._normalize_proprio(sampled_proprio).astype(np.float32)

        actions = torch.from_numpy(episode["actions"]).float()
        last_action = actions[-1:]
        padding_steps = max(500, self.chunk_len + self.group_loss_window * self.action_overcollect_ratio)
        extended_actions = torch.cat([actions, last_action.repeat(padding_steps, 1)], dim=0)

        start_idx = self.action_overcollect_ratio * sample_idx
        sampled_action_chunks: List[torch.Tensor] = []
        for j in range(self.group_loss_window):
            offset = j * self.action_overcollect_ratio
            chunk_start = start_idx + offset
            chunk_end = chunk_start + self.chunk_len
            sampled_action_chunks.append(extended_actions[chunk_start:chunk_end])

        sampled_actions = torch.stack(sampled_action_chunks, dim=0).cpu().numpy().astype(np.float32)
        sampled_actions = self._normalize_action(sampled_actions).astype(np.float32)

        sample = {
            "cam_global": sampled_top_tensor,
            "proprioceptive": torch.from_numpy(sampled_proprio).float(),
            "action_future": torch.from_numpy(sampled_actions).float(),
            "task_emb": torch.from_numpy(episode["task_emb"]).float(),
        }
        return sample, task_name, episode_id, sample_idx

    def __iter__(self):
        while True:
            yield self._sample()
