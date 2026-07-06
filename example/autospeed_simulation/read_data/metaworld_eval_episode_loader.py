from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class MetaWorldEvalEpisodeLoader:
    def __init__(
        self,
        eval_data_paths: List[str],
        stats_path: str | Path,
        img_size: List[int],
        num_queries: int,
        speed_range: List[float],
        action_overcollect_ratio: int = 1,
        proprio_dim: int = 9,
    ) -> None:
        self.eval_data_paths = eval_data_paths
        self.stats_path = Path(stats_path)
        self.img_size = img_size
        self.num_queries = int(num_queries)
        self.a_max = float(speed_range[1])
        self.action_overcollect_ratio = int(action_overcollect_ratio)
        self.proprio_dim = int(proprio_dim)

        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_path}")

        with h5py.File(self.stats_path, "r") as f:
            self.action_min = f["action_min"][:].astype(np.float32)
            self.action_max = f["action_max"][:].astype(np.float32)
            self.proprio_min = f["proprio_min"][:].astype(np.float32)
            self.proprio_max = f["proprio_max"][:].astype(np.float32)

        self.chunk_len = int(
            math.ceil(self.num_queries * self.a_max * self.action_overcollect_ratio)
        )

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.resize_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        self.episodes: List[Dict] = []
        print("[MetaWorldEvalEpisodeLoader] loading eval episodes...")
        for eval_data_path in tqdm(self.eval_data_paths):
            self.episodes.append(self._load_episode(eval_data_path))

    def _normalize_action(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.action_min) / (self.action_max - self.action_min + 1e-5) - 1.0

    def _normalize_proprio(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.proprio_min) / (self.proprio_max - self.proprio_min + 1e-5) - 1.0

    def _load_episode(self, eval_data_path: str) -> Dict:
        with h5py.File(eval_data_path, "r") as f:
            actions = f["action"][:].astype(np.float32)
            top_camera = f["observations/images/top"][:]
            proprio = f["observations/qpos"][:, : self.proprio_dim].astype(np.float32)
            task_emb = f["task_emb"][:].astype(np.float32)

        if actions.shape[1] != 4:
            raise ValueError(
                f"Expected action dim 4, got {actions.shape} in {eval_data_path}"
            )
        if not (len(actions) == len(proprio) == len(top_camera)):
            raise ValueError(
                "Length mismatch in episode "
                f"{eval_data_path}: len(action)={len(actions)}, len(qpos)={len(proprio)}, len(images)={len(top_camera)}"
            )

        num_frames = len(proprio)
        h, w = top_camera.shape[1], top_camera.shape[2]
        use_resize = h != self.img_size[0] or w != self.img_size[1]
        aug = self.resize_aug if use_resize else self.aug

        top_camera_tensor = torch.stack([aug(img) for img in top_camera])

        proprio = self._normalize_proprio(proprio).astype(np.float32)
        proprio_tensor = torch.from_numpy(proprio).float()

        actions_raw = torch.from_numpy(actions).float()
        last_action = actions_raw[-1:]
        padded_actions = torch.cat(
            [actions_raw, last_action.repeat(self.chunk_len + self.action_overcollect_ratio, 1)],
            dim=0,
        )

        action_chunks = []
        for t in range(num_frames):
            start_idx = self.action_overcollect_ratio * t
            end_idx = start_idx + self.chunk_len
            action_chunks.append(padded_actions[start_idx:end_idx])

        action_chunks = torch.stack(action_chunks, dim=0).cpu().numpy().astype(np.float32)
        action_chunks = self._normalize_action(action_chunks).astype(np.float32)
        action_chunks_tensor = torch.from_numpy(action_chunks).float()

        task_name = os.path.basename(os.path.dirname(eval_data_path))
        match = re.search(r"episode_(\d+)\.hdf5$", eval_data_path)
        episode_id = int(match.group(1)) if match else -1

        obs = {
            "cam_global": top_camera_tensor,
            "proprioceptive": proprio_tensor,
        }

        return {
            "num_frames": num_frames,
            "task_emb": task_emb,
            "obs": obs,
            "action_chunks": action_chunks_tensor,
            "task_name": task_name,
            "episode_id": episode_id,
        }

    def sample_from_filename(self, eval_data_path: str) -> Dict:
        index = self.eval_data_paths.index(eval_data_path)
        return self.episodes[index]
