import os
import sys
import time
import gc
import warnings
import argparse
import csv

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)

import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import OmegaConf, open_dict

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import utils.agent_utils as utils
from utils.logger_ddp import Logger
from utils.replay_buffer import make_expert_replay_loader
from utils.video import VideoRecorder
from utils.nonlinear_temporal_agg import NonlinearTemporalAgg

torch.backends.cudnn.benchmark = True

def _crop14(img):
    _, h, w = img.shape
    nh, nw = (h // 14) * 14, (w // 14) * 14
    if nh == h and nw == w:
        return img
    top, left = (h - nh) // 2, (w - nw) // 2
    return transforms.functional.crop(img, top, left, nh, nw)

class WorkspaceIL:
    def __init__(self, args):
        utils.set_seed_everywhere(42)
        self.args = args
        self._closed = False

        self.ckpt_dir = os.path.dirname(os.path.dirname(self.args.ckpt_path))
        self.cfg_path = os.path.join(self.ckpt_dir, "full_config.yaml")
        ckpt_name = Path(self.args.ckpt_path).stem
        self.eval_dir = Path(self.ckpt_dir) / f"eval_{ckpt_name}" / time.strftime("%Y%m%d%H%M")
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        print(f"eval dir: {self.eval_dir}")

        self.cfg = OmegaConf.load(self.cfg_path)
        print(f"Loaded config from {self.cfg_path}")

        self.save_video = False
        self.temporal_agg = False
        self.use_nta = False
        self.num_eval_episodes = int(args.num_eval_episodes)
        self.max_timesteps = int(args.max_timesteps)
        self.start_idx = int(args.start_idx)
        self.end_idx = args.end_idx

        self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.stats = {}
        self.aggregator = NonlinearTemporalAgg()

        print('Loading expert dataset...')
        dataset_iterable = hydra.utils.call(self.cfg.dataloader.bc_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, batch_size=1, num_workers=1
        )
        print(f'Dataset loaded, max_episode_len={self.expert_replay_loader.dataset._max_episode_len}')

        with open_dict(self.cfg):
            self.cfg.suite.task_make_fn.max_episode_len = self.expert_replay_loader.dataset._max_episode_len
            if hasattr(self.expert_replay_loader.dataset, '_max_state_dim'):
                self.cfg.suite.task_make_fn.max_state_dim = self.expert_replay_loader.dataset._max_state_dim

        print('Creating environment...')
        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        print(f'Created {len(self.env)} environments')

        print('Creating agent...')
        self.agent = hydra.utils.instantiate(self.cfg.agent).to(self.device)
        print('Agent created')

        self.logger = Logger(self.eval_dir, use_tb=False, rank=0)
        self.video_recorder = VideoRecorder(self.eval_dir if self.save_video else None)
        try:
            self.run_eval()
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True

        envs = getattr(self, "env", None)
        if envs is not None:
            for env_idx, env in enumerate(envs):
                close_fn = getattr(env, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception as exc:
                        print(f"[cleanup] env {env_idx} close failed: {exc!r}")
            self.env = []

        logger = getattr(self, "logger", None)
        finish_fn = getattr(logger, "finish", None)
        if callable(finish_fn):
            try:
                finish_fn()
            except Exception:
                pass

        gc.collect()

    def load_snapshot(self, ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, weights_only=False)
        agent_payload = {k: v for k, v in payload.items() if k not in self.__dict__}
        self.agent.load_snapshot(agent_payload)
        print("Checkpoint loaded")

    def load_stats(self):
        stats_path = os.path.join(self.ckpt_dir, "stats.hdf5")
        if not os.path.exists(stats_path):
            ds = self.expert_replay_loader.dataset
            if hasattr(ds, 'action_mins') and hasattr(ds, 'action_maxs'):
                self.stats["min"] = ds.action_mins
                self.stats["max"] = ds.action_maxs
                print("Loaded stats from dataset")
                return
            print(f"Stats file not found: {stats_path}, using default normalization")
            return
        with h5py.File(stats_path, 'r') as f:
            self.stats["min"] = f['min'][:]
            self.stats["max"] = f['max'][:]
        print(f"Loaded stats from {stats_path}")

    def denorm_action(self, action):
        if "min" not in self.stats or "max" not in self.stats:
            return action
        min_t = torch.tensor(self.stats["min"], device=action.device, dtype=action.dtype)
        max_t = torch.tensor(self.stats["max"], device=action.device, dtype=action.dtype)
        return (action + 1.0) / 2.0 * (max_t - min_t) + min_t

    def get_obs_dict(self, time_step):
        obs = time_step.observation
        pixel_keys = self.cfg.suite.pixel_keys
        agent_keys = getattr(self.cfg.agent, "pixel_keys", pixel_keys)
        img_size = list(self.cfg.suite.img_size)                   

        obs_dict = {}
        for env_k, agent_k in zip(pixel_keys, agent_keys):
            img = obs.get(env_k) or obs.get("pixels" if env_k == "high_camera" else "pixels_egocentric")
            if img is None:
                raise KeyError(f"Missing key {env_k}, available: {list(obs.keys())}")

            img = torch.from_numpy(img).float()
            if img.shape[-1] in (1, 3):              
                img = img.permute(2, 0, 1)
            img = img / 255.0

            img = img.unsqueeze(0)                
            img = F.interpolate(img, size=img_size, mode='bilinear', align_corners=False)
            img = img.squeeze(0)             

            img = _crop14(img)
            obs_dict[agent_k] = img.unsqueeze(0).to(self.device)

        proprio_key = self.cfg.suite.proprio_key
        proprio = torch.from_numpy(obs[proprio_key]).float().unsqueeze(0).to(self.device)
            
        lang = torch.as_tensor(obs.get("task_emb", np.zeros(384)), dtype=torch.float32)
        lang = lang.unsqueeze(0).unsqueeze(0).to(self.device)

        return obs_dict, proprio, lang

    def run_eval(self):
        self.load_snapshot(self.args.ckpt_path)
        self.load_stats()
        self.agent.train(False)

        num_future = int(getattr(self.cfg.agent, "num_queries", 16) or 16)
        print(f'num_future: {num_future}')
        if self.end_idx is None:
            self.end_idx = num_future
        else:
            self.end_idx = int(self.end_idx)

        if self.start_idx < 0 or self.end_idx <= self.start_idx or self.end_idx > num_future:
            raise ValueError(
                f"Invalid start/end idx: start={self.start_idx}, end={self.end_idx}, num_future={num_future}. "
                "Require 0 <= start < end <= num_future."
            )
        if self.temporal_agg:
            start_idx = 0
            end_idx = num_future
            query_frequency = 1
        else:
            start_idx = self.start_idx
            end_idx = self.end_idx
            query_frequency = end_idx - start_idx
        query_chunk = end_idx - start_idx
        print(
            f"eval config: temporal_agg={self.temporal_agg}, use_nta={self.use_nta}, "
            f"start_idx={start_idx}, end_idx={end_idx}, query_frequency={query_frequency}, "
            f"max_timesteps={self.max_timesteps}"
        )

        all_success = []
        all_rewards = []
        all_success_steps = []
        rollout_records = []
        env_summary_records = []

        for env_idx in range(self.envs_till_idx):
            print(f"\n=== Evaluating env {env_idx}: {self.task_descriptions[env_idx]} ===")
            env_success = []
            env_rewards = []
            env_steps = []
            env_success_steps = []

            for ep in range(self.num_eval_episodes):
                print(f'episode {ep+1} / {self.num_eval_episodes}')
                self.aggregator.reset()
                time_step = self.env[env_idx].reset()
                self.video_recorder.init(self.env[env_idx], enabled=self.save_video)

                total_reward = 0
                current_speed = 1.0
                action_chunk = None
                all_time_actions = None
                for t in range(self.max_timesteps):                   
                    with torch.no_grad():
                        if (t % query_frequency == 0) or (action_chunk is None):
                            obs_dict, proprio, lang = self.get_obs_dict(time_step)
                            action_chunk, pred_a = self.agent.act(obs_dict, proprio, lang, None)
                            action_chunk = action_chunk[:, start_idx:end_idx]
                            current_speed = float(pred_a.item())
                            if self.temporal_agg and all_time_actions is None:
                                all_time_actions = torch.zeros(
                                    (
                                        self.max_timesteps,
                                        self.max_timesteps + query_chunk,
                                        action_chunk.shape[-1],
                                    ),
                                    device=self.device,
                                    dtype=action_chunk.dtype,
                                )
                        if self.temporal_agg:
                            if self.use_nta:
                                raw_action = self.aggregator.record_and_get_current_actions(
                                    action_chunk.squeeze(0), current_speed, t
                                )
                            else:
                                all_time_actions[[t], t : t + query_chunk] = action_chunk
                                actions_for_curr_step = all_time_actions[:, t]
                                actions_populated = torch.all(
                                    actions_for_curr_step != 0, dim=1
                                )
                                actions_for_curr_step = actions_for_curr_step[
                                    actions_populated
                                ]
                                k = 0.01
                                exp_weights = np.exp(
                                    -k * np.arange(len(actions_for_curr_step))
                                )
                                exp_weights = exp_weights / exp_weights.sum()
                                exp_weights = (
                                    torch.from_numpy(exp_weights)
                                    .to(self.device)
                                    .unsqueeze(dim=1)
                                )
                                raw_action = (actions_for_curr_step * exp_weights).sum(
                                    dim=0, keepdim=True
                                )
                        else:
                            offset = t % query_frequency
                            action_idx = min(offset, action_chunk.shape[1] - 1)
                            raw_action = action_chunk[:, action_idx]

                        action = self.denorm_action(raw_action)
                        action = action.cpu().numpy()

                    time_step = self.env[env_idx].step(action)
                    if self.save_video:
                        overlay_text = f"idx:{t}\\na:{current_speed:.2f}"
                        self.video_recorder.record(
                            self.env[env_idx], overlay_text=overlay_text
                        )
                    total_reward += time_step.reward

                    if time_step.observation.get("goal_achieved", False):
                        break

                success = time_step.observation.get("goal_achieved", False)
                steps = t + 1
                env_success.append(success)
                env_rewards.append(total_reward)
                env_steps.append(steps)
                if success:
                    env_success_steps.append(steps)

                if self.save_video:
                    self.video_recorder.save(f"env{env_idx}_ep{ep}.mp4")

                print(f"  Episode {ep}: reward={total_reward:.2f}, success={success}, steps={steps}")
                rollout_records.append(
                    {
                        "env_idx": int(env_idx),
                        "episode_idx": int(ep),
                        "task_description": str(self.task_descriptions[env_idx]),
                        "reward": float(total_reward),
                        "success": bool(success),
                        "steps": int(steps),
                    }
                )

            sr = np.mean(env_success)
            all_success.extend(env_success)
            all_rewards.extend(env_rewards)
            all_success_steps.extend(env_success_steps)
            print(f"  Env {env_idx} success rate: {sr:.2%}")
            env_summary_records.append(
                {
                    "env_idx": int(env_idx),
                    "task_description": str(self.task_descriptions[env_idx]),
                    "success_rate": float(sr),
                    "avg_reward": float(np.mean(env_rewards)) if len(env_rewards) > 0 else 0.0,
                    "avg_steps": float(np.mean(env_steps)) if len(env_steps) > 0 else 0.0,
                    "avg_success_len": (
                        float(np.mean(env_success_steps)) if len(env_success_steps) > 0 else 0.0
                    ),
                }
            )

        total_sr = np.mean(all_success)
        total_reward = np.mean(all_rewards)
        avg_env_avg_steps = (
            float(np.mean([record["avg_steps"] for record in env_summary_records]))
            if len(env_summary_records) > 0
            else 0.0
        )
        avg_success_len = (
            float(np.mean(all_success_steps))
            if len(all_success_steps) > 0
            else 0.0
        )
        result_str = (
            f"Total Success Rate: {total_sr:.2%}\n"
            f"Average Reward: {total_reward:.2f}\n"
            f"Average of Per-Env Avg Steps: {avg_env_avg_steps:.1f}\n"
            f"avg_success_len: {avg_success_len:.1f}\n"
        )
        print(f"\n{result_str}")

        with open(self.eval_dir / "result.txt", "w") as f:
            f.write(result_str)
            f.write(f"\nCheckpoint: {self.args.ckpt_path}\n")
            f.write(
                f"temporal_agg: {self.temporal_agg}\nuse_nta: {self.use_nta}\n"
                f"self.start_idx: {self.start_idx}\nself.end_idx: {self.end_idx}\n"
                f"max_timesteps: {self.max_timesteps}\n"
            )
            f.write("\n[Per-Env Summary]\n")
            for env_info in env_summary_records:
                f.write(
                    f"env={env_info['env_idx']}, task={env_info['task_description']}, "
                    f"success_rate={env_info['success_rate']:.2%}, "
                    f"avg_reward={env_info['avg_reward']:.2f}, "
                    f"avg_steps={env_info['avg_steps']:.1f}, "
                    f"avg_success_len={env_info['avg_success_len']:.1f}\n"
                )

        metrics_csv_path = self.eval_dir / "rollout_metrics.csv"
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "env_idx",
                    "episode_idx",
                    "task_description",
                    "reward",
                    "success",
                    "steps",
                ],
            )
            writer.writeheader()
            for record in rollout_records:
                writer.writerow(record)
        print(f"Saved rollout metrics to {metrics_csv_path}")

        with self.logger.log_and_dump_ctx(0, ty="eval") as log:
            log("success", total_sr)
            log("episode_reward", total_reward)
            log("avg_env_avg_steps", avg_env_avg_steps)
            log("avg_success_len", avg_success_len)
            for i, sr in enumerate(all_success):
                log(f"success_env{i//self.num_eval_episodes}", sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LIBERO checkpoint in environment rollout.")
    parser.add_argument("--ckpt-path", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--num-eval-episodes", type=int, default=50)
    parser.add_argument("--max-timesteps", type=int, default=500)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()
    WorkspaceIL(args)