                      

import argparse
import os
import re
import sys
import time

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from tqdm import tqdm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import utils.agent_utils as utils
from utils.nonlinear_temporal_agg import NonlinearTemporalAgg
from suite.metaworld_wrapper import make as make_metaworld_env, resolve_metaworld_task_name

try:
    import cv2
except ImportError:
    cv2 = None

class WorkspaceEvalMetaWorld:
    def __init__(self, args):
        self.args = args
        utils.set_seed_everywhere(args.seed)

        self.num_rollouts = int(args.num_rollouts)
        self.save_video = bool(args.save_video)
        self.camera_name = "corner2"

        self.model_flip_vertical = True
        self.model_flip_horizontal = True

        self.video_flip_vertical = True
        self.video_flip_horizontal = False

        self.ckpt_path = Path(args.ckpt_path).expanduser().resolve()
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        self.run_dir = self.ckpt_path.parent.parent
        self.cfg_path = self.run_dir / "full_config.yaml"
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {self.cfg_path}")
        self.cfg = OmegaConf.load(self.cfg_path)

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

        self.agent = hydra.utils.instantiate(self.cfg.agent).to(self.device)
        self._load_snapshot()
        self.agent.train(False)

        self.action_min, self.action_max, self.proprio_min, self.proprio_max = self._load_stats()
        self.task_name = self._infer_task_name()
        if args.task_name:
            self.task_name = resolve_metaworld_task_name(args.task_name)
        print(self.task_name)
        self.data_root = Path(str(self.cfg.dataloader.bc_dataset.get("path", ""))).expanduser()
        self.max_timesteps = (
            int(args.max_timesteps)
            if args.max_timesteps is not None
            else int(self._infer_task_max_timesteps(self.task_name))
        )
        cfg_img_size = self.cfg.suite.get("img_size", [112, 112])
        self.eval_img_size = tuple(int(x) for x in cfg_img_size)

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.resize_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.eval_img_size),
            transforms.ToTensor(),
        ])

        self.env = make_metaworld_env(
            task_name=self.task_name,
            seed=args.seed,
            camera_name=self.camera_name,
        )
        self.prompt = self._build_task_prompt()
        self.aggregator = NonlinearTemporalAgg()

        ckpt_stem = self.ckpt_path.stem
        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.eval_dir = self.run_dir / f"eval_online_{ckpt_stem}" / timestamp
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"Eval dir: {self.eval_dir}")
        print(f"Task: {self.task_name}")
        print(f"Camera: {self.camera_name}")
        print(f"Checkpoint: {self.ckpt_path}")
        print(f"Max timesteps: {self.max_timesteps}")

    def _load_snapshot(self):
        with self.ckpt_path.open("rb") as f:
            payload = torch.load(f, map_location=self.device, weights_only=False)
        self.agent.load_snapshot(payload)

    def _load_stats(self):
        stats_path = self.run_dir / "stats.hdf5"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.hdf5 not found: {stats_path}")
        with h5py.File(stats_path, "r") as f:
            action_min = torch.from_numpy(f["action_min"][:].astype(np.float32)).to(self.device)
            action_max = torch.from_numpy(f["action_max"][:].astype(np.float32)).to(self.device)
            proprio_min = torch.from_numpy(f["proprio_min"][:].astype(np.float32)).to(self.device)
            proprio_max = torch.from_numpy(f["proprio_max"][:].astype(np.float32)).to(self.device)
        return action_min, action_max, proprio_min, proprio_max

    def _infer_task_name(self):
        tasks_cfg = self.cfg.suite.task.tasks
        tasks = OmegaConf.to_container(tasks_cfg, resolve=True)
        return str(next(iter(tasks[0].values()))[0])

    def _build_task_prompt(self):
        eval_path = self._select_prompt_path()
        if eval_path is not None and os.path.exists(eval_path):
            with h5py.File(eval_path, "r") as f:
                if "task_emb" in f:
                    task_emb = f["task_emb"][:].astype(np.float32)
                    print(
                        f"Loaded task_emb from: {eval_path} "
                        f"(norm={float(np.linalg.norm(task_emb)):.4f})"
                    )
                    if hasattr(self.env, "task_emb"):
                        self.env.task_emb = task_emb.copy()
                    return torch.from_numpy(task_emb).to(self.device).unsqueeze(0).unsqueeze(0)
        print(f"No valid task_emb file found for task={self.task_name}; fallback to zero prompt.")
        task_emb = np.zeros((384,), dtype=np.float32)
        if hasattr(self.env, "task_emb"):
            self.env.task_emb = task_emb.copy()
        return torch.from_numpy(task_emb).to(self.device).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _episode_sort_key(path_obj: Path):
        match = re.search(r"episode_(\d+)\.hdf5$", path_obj.name)
        episode_id = int(match.group(1)) if match else 10**12
        return (str(path_obj.parent), episode_id, path_obj.name)

    @staticmethod
    def _normalize_task_name(name):
        text = str(name).strip()
        if text == "":
            return ""
        try:
            return resolve_metaworld_task_name(text).lower()
        except Exception:
            return text.lower()

    def _path_matches_task(self, file_path: Path, task_key: str):
        parent_key = self._normalize_task_name(file_path.parent.name)
        if parent_key == task_key:
            return True
        if not file_path.exists():
            return False
        try:
            with h5py.File(file_path, "r") as f:
                attr_name = f.attrs.get("task_name", "")
        except Exception:
            return False
        if isinstance(attr_name, bytes):
            attr_name = attr_name.decode("utf-8", errors="ignore")
        attr_key = self._normalize_task_name(attr_name)
        return attr_key == task_key

    def _select_prompt_path(self):
        task_key = self._normalize_task_name(self.task_name)
        eval_paths = self.cfg.suite.task.get("eval_data_paths", None)
        candidates = [Path(str(p)).expanduser() for p in list(eval_paths)] if eval_paths else []

        for p in candidates:
            if self._path_matches_task(p, task_key):
                return str(p)

        data_root = self.cfg.dataloader.bc_dataset.get("path", None)
        if data_root is not None:
            root = Path(str(data_root)).expanduser()
            if root.exists():
                seen = set()
                task_dirs = []
                for name in {str(self.task_name), task_key}:
                    if name:
                        d = root / name
                        if d.exists() and d.is_dir() and d not in seen:
                            seen.add(d)
                            task_dirs.append(d)
                for d in root.iterdir():
                    if d.is_dir() and self._normalize_task_name(d.name) == task_key and d not in seen:
                        seen.add(d)
                        task_dirs.append(d)

                for task_dir in task_dirs:
                    episodes = sorted(task_dir.glob("episode_*.hdf5"), key=self._episode_sort_key)
                    if len(episodes) > 0:
                        return str(episodes[0])

                for p in sorted(root.rglob("episode_*.hdf5"), key=self._episode_sort_key):
                    if self._path_matches_task(p, task_key):
                        return str(p)

        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def _infer_task_max_timesteps(self, task_name: str) -> int:
        if not self.data_root.exists():
            print(f"Dataset root not found: {self.data_root}, fallback max_timesteps=500")
            return 500

        task_key = self._normalize_task_name(task_name)
        task_dirs = []

        direct_dir = self.data_root / str(task_name)
        if direct_dir.exists() and direct_dir.is_dir():
            task_dirs.append(direct_dir)

        for d in self.data_root.iterdir():
            if d.is_dir() and self._normalize_task_name(d.name) == task_key and d not in task_dirs:
                task_dirs.append(d)

        max_len = 0
        checked_files = 0
        for task_dir in task_dirs:
            for ep_path in task_dir.glob("episode_*.hdf5"):
                try:
                    with h5py.File(ep_path, "r") as f:
                        if "action" not in f:
                            continue
                        ep_len = int(f["action"].shape[0])
                    checked_files += 1
                    if ep_len > max_len:
                        max_len = ep_len
                except Exception:
                    continue

        if max_len <= 0:
            print(f"Could not infer max timesteps for task={task_name}, fallback max_timesteps=500")
            return 500
        print(
            f"Inferred max_timesteps={max_len} for task={task_name} "
            f"from {checked_files} episodes"
        )
        return max_len

    def _normalize_proprio(self, proprio_np):
        proprio = torch.from_numpy(proprio_np.astype(np.float32)).to(self.device)
        return 2.0 * (proprio - self.proprio_min) / (self.proprio_max - self.proprio_min + 1e-5) - 1.0

    def _denorm_action(self, action):
        return (action + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min

    def _align_model_frame(self, img):
        if self.model_flip_vertical:
            img = img[::-1, :, :]
        if self.model_flip_horizontal:
            img = img[:, ::-1, :]
        if self.model_flip_vertical or self.model_flip_horizontal:
            return np.ascontiguousarray(img)
        return img

    def _align_video_frame(self, img):
        if self.video_flip_vertical:
            img = img[::-1, :, :]
        if self.video_flip_horizontal:
            img = img[:, ::-1, :]
        if self.video_flip_vertical or self.video_flip_horizontal:
            return np.ascontiguousarray(img)
        return img

    def _obs_to_agent_inputs(self, obs):
        img = self._align_model_frame(obs["cam_global"])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[0], img.shape[1]
        use_resize = h != self.eval_img_size[0] or w != self.eval_img_size[1]
        aug = self.resize_aug if use_resize else self.aug
        img_t = aug(img).unsqueeze(0).to(self.device)

        obs_dict = {"cam_global": img_t}
        proprio_t = self._normalize_proprio(obs["proprioceptive"]).unsqueeze(0)
        return obs_dict, proprio_t

    def _save_video(self, frames_rgb, speed_list, query_frequency, save_path):
        if len(frames_rgb) == 0:
            return
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required when save_video=True")
        h, w = frames_rgb[0].shape[:2]
        writer = cv2.VideoWriter(
            str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), float(self.args.video_fps), (w, h)
        )
        for idx, frame in enumerate(frames_rgb):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(frame_bgr, f"idx:{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if len(speed_list) > 0:
                speed_idx = min(idx // query_frequency, len(speed_list) - 1)
                cv2.putText(
                    frame_bgr,
                    f"a:{speed_list[speed_idx]:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
            writer.write(frame_bgr)
        writer.release()

    def _save_speed_plot(self, speed_list, query_frequency, rollout_id):
        if len(speed_list) == 0:
            return
        plt.figure(figsize=(10, 4))
        x = np.arange(len(speed_list)) * query_frequency
        plt.plot(x, speed_list, linestyle="--", marker="o", markersize=3, linewidth=1.5)
        plt.xlabel("timestep")
        plt.ylabel("pred speed a")
        plt.grid(True, alpha=0.3)
        plt.title(f"Rollout {rollout_id} Predicted Speed")
        plt.tight_layout()
        plt.savefig(self.eval_dir / f"rollout{rollout_id}_a.png", dpi=120)
        plt.close()

    def run(self):
        num_queries = int(self.cfg.agent.num_queries)
        max_timesteps = int(self.max_timesteps)
        if self.args.temporal_agg:
            query_frequency = 1
            start_idx = 0
            end_idx = num_queries
        else:
            start_idx = 0
            end_idx = num_queries
            query_frequency = end_idx - start_idx
        query_chunk = end_idx - start_idx
        print(
            f"eval config: temporal_agg={self.args.temporal_agg}, use_nta={self.args.use_nta}, "
            f"start_idx={start_idx}, end_idx={end_idx}, query_frequency={query_frequency}, "
            f"max_timesteps={max_timesteps}"
        )

        all_success = []
        all_returns = []
        all_lengths = []

        for rollout_id in range(self.num_rollouts):
            self.aggregator.reset()
            obs = self.env.reset()
            frames = []
            rewards = []
            speed_list = []
            action_chunk = None
            current_speed = 1.0
            all_time_actions = None

            for t in tqdm(range(max_timesteps), desc=f"Rollout {rollout_id}", leave=False):
                frames.append(self._align_video_frame(obs["cam_global"]))
                obs_dict, proprio = self._obs_to_agent_inputs(obs)

                with torch.no_grad():
                    if (t % query_frequency == 0) or (action_chunk is None):
                        action_chunk, pred_a = self.agent.act(obs_dict, proprio, self.prompt, None)
                        action_chunk = self._denorm_action(action_chunk)[:, start_idx:end_idx, :]
                        current_speed = float(pred_a[0].item())
                        speed_list.append(current_speed)
                        if self.args.temporal_agg and all_time_actions is None:
                            all_time_actions = torch.zeros(
                                (
                                    max_timesteps,
                                    max_timesteps + query_chunk,
                                    action_chunk.shape[-1],
                                ),
                                device=self.device,
                                dtype=action_chunk.dtype,
                            )

                    if self.args.temporal_agg:
                        if self.args.use_nta:
                            raw_action = self.aggregator.record_and_get_current_actions(
                                action_chunk.squeeze(0), current_speed, t
                            )
                        else:
                            all_time_actions[[t], t : t + query_chunk] = action_chunk
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, dim=1
                            )
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        offset = t % query_frequency
                        action_idx = min(offset, action_chunk.shape[1] - 1)
                        raw_action = action_chunk[:, action_idx]

                action_np = raw_action.squeeze(0).detach().cpu().numpy().astype(np.float32)
                obs, reward, done, info = self.env.step(action_np)
                rewards.append(float(reward))

                success = bool(obs.get("goal_achieved", False) or info.get("success", False))
                done_bool = bool(np.asarray(done).any())
                if success or done_bool:
                    break

            episode_return = float(np.sum(rewards))
            episode_len = len(rewards)
            final_success = bool(obs.get("goal_achieved", False))

            all_success.append(final_success)
            all_returns.append(episode_return)
            all_lengths.append(episode_len)

            print(
                f"Rollout {rollout_id}: success={final_success}, "
                f"return={episode_return:.4f}, len={episode_len}"
            )

            if self.save_video:
                self._save_video(
                    frames_rgb=frames,
                    speed_list=speed_list,
                    query_frequency=query_frequency,
                    save_path=self.eval_dir / f"rollout{rollout_id}.mp4",
                )
            self._save_speed_plot(speed_list, query_frequency, rollout_id)

        summary = {
            "num_rollouts": int(self.num_rollouts),
            "task_name": self.task_name,
            "checkpoint": str(self.ckpt_path),
            "success_rate": float(np.mean(all_success)),
            "avg_return": float(np.mean(all_returns)),
            "avg_len": float(np.mean(all_lengths)),
            "avg_success_len": (
                float(np.mean([l for l, s in zip(all_lengths, all_success) if s]))
                if any(all_success)
                else 0.0
            ),
            "max_timesteps": int(max_timesteps),
            "temporal_agg": bool(self.args.temporal_agg),
            "use_nta": bool(self.args.use_nta),
        }

        print("\n=== Eval Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

        result_path = self.eval_dir / "result.txt"
        with result_path.open("w", encoding="utf-8") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write(f"all_success: {all_success}\n")
            f.write(f"all_returns: {all_returns}\n")
            f.write(f"all_lengths: {all_lengths}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-video", dest="save_video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--temporal-agg", action="store_true", default=False)
    parser.add_argument("--use-nta", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    worker = WorkspaceEvalMetaWorld(args)
    worker.run()


if __name__ == "__main__":
    main()
