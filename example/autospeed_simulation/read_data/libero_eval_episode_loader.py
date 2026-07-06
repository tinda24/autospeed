import pickle as pkl
from pathlib import Path
import math

import torch
import torchvision.transforms as transforms
import numpy as np
import h5py                                    
from read_data.libero_ddp import _crop14, _load_stats

class LiberoEvalEpisodeLoader:

    def __init__(
        self,
        dataset_root,
        suite,
        stats_path,
        img_size,
        num_queries,
        action_overcollect_ratio,
        speed_range,
    ):
        self.dataset_root = Path(dataset_root)
        self.suite = suite
        self.stats_path = Path(stats_path)
        self.img_size = img_size if img_size is not None else [128, 128]
        self.num_queries = num_queries
        self.action_overcollect_ratio = action_overcollect_ratio
        self.a_min, self.a_max = speed_range[0], speed_range[1]

                                                                  
        self._future_window = int(math.ceil(
            self.num_queries * self.a_max * self.action_overcollect_ratio
        ))
                                  
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_path}")

        with h5py.File(self.stats_path, 'r') as f:
            self.action_mins = np.asarray(f['min'][:], dtype=np.float32)
            self.action_maxs = np.asarray(f['max'][:], dtype=np.float32)
                      
        self._pkl_cache = {}

    def _preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            x = img.permute(2, 0, 1) if img.ndim == 3 and img.shape[-1] in (1, 3, 4) else img
            x = x.unsqueeze(0) if x.ndim == 2 else x
            x = x.float() / 255.0 if x.max() > 1.0 else x.float()
        else:
            x = transforms.functional.to_tensor(img)
        x = transforms.functional.resize(x, self.img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        return _crop14(x)

    def _normalize_actions(self, x):
        x = np.asarray(x, dtype=np.float32)
        ranges = np.maximum(self.action_maxs - self.action_mins, 1e-6)
        return np.clip((x - self.action_mins) / ranges * 2.0 - 1.0, -1.0, 1.0)

    def _load_pkl(self, task_name: str):
        if task_name in self._pkl_cache:
            return self._pkl_cache[task_name]

        pkl_path = self.dataset_root / self.suite / f"{task_name}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Task pkl not found: {pkl_path}")

        data = pkl.load(open(str(pkl_path), "rb"))
        self._pkl_cache[task_name] = data
        return data

    def sample_from_spec(self, task_name: str, demo_id: int) -> dict:
        data = self._load_pkl(task_name)

        observations = data["observations"][demo_id]
        actions = data["actions"][demo_id]
        task_emb = torch.as_tensor(data["task_emb"], dtype=torch.float32)
                  
        num_frames = len(observations["pixels"])
           
        high_camera = torch.stack([
            self._preprocess_image(img) for img in observations["pixels"]
        ])
        right_camera = torch.stack([
            self._preprocess_image(img) for img in observations["pixels_egocentric"]
        ])
                                      
        proprioceptive = np.concatenate([
            observations["joint_states"],
            observations["gripper_states"]
        ], axis=-1)
        proprioceptive = torch.from_numpy(proprioceptive).float()

                                
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]
                                
        back_pad = np.repeat(actions[-1:], self._future_window, axis=0)
        padded_actions = np.concatenate([actions, back_pad], axis=0)
                
        padded_actions = self._normalize_actions(padded_actions)
        padded_actions = torch.from_numpy(padded_actions).float()
                            
        action_chunks = []
        for t in range(num_frames):
            start_idx = t * self.action_overcollect_ratio
            end_idx = start_idx + self._future_window
            chunk = padded_actions[start_idx:end_idx]
            action_chunks.append(chunk)

        action_chunks = torch.stack(action_chunks, dim=0)

        obs = {
            "high_camera": high_camera,
            "right_camera": right_camera,
            "proprioceptive": proprioceptive,
        }

        return {
            "num_frames": num_frames,
            "task_emb": task_emb,
            "obs": obs,
            "action_chunks": action_chunks,
            "task_name": task_name,
            "episode_id": demo_id,
        }
