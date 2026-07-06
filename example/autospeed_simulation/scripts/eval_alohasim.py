import torch
import numpy as np
import os
import sys
import time
import yaml
import cv2

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pickle
import math
import h5py
import argparse
import hydra
import inspect
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
import torchvision.transforms as transforms

from suite.act.constants import DT
from suite.act.constants import PUPPET_GRIPPER_JOINT_OPEN
from suite.act.constants import SIM_TASK_CONFIGS
from suite.act.act_utils import load_data                  
from suite.act.act_utils import (
    sample_box_pose,
    sample_insertion_pose,
)                   
from suite.act.act_utils import set_seed                    
from suite.act.visualize_episodes import save_videos
from suite.act.sim_env import BOX_POSE, make_sim_env

from utils.nonlinear_temporal_agg import NonlinearTemporalAgg

def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)

def save_videos_with_frame_idx(
    video, dt, video_path, query_frequency=1, speed_list=None
):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    cam_names = list(video[0].keys())
    h, w, _ = video[0][cam_names[0]].shape
    w = w * len(cam_names)
    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), int(1 / dt), (w, h)
    )
    for ts, image_dict in enumerate(video):
        images = np.concatenate(
            [image_dict[c][:, :, [2, 1, 0]] for c in cam_names], axis=1
        ).copy()
        cv2.putText(
            images, f"idx:{ts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        if speed_list is not None:
            speed_idx = ts // query_frequency
            if speed_idx < len(speed_list):
                cv2.putText(
                    images,
                    f"a:{speed_list[speed_idx]:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
        out.write(images)
    out.release()
    print(f"Saved video to: {video_path}")

class WorkspaceIL:
    def __init__(self, args):
        self.args = args
        set_seed(2)
        self.ckpt_path = str(Path(self.args.ckpt_path).expanduser().resolve())

        self.ckpt_dir = os.path.dirname(os.path.dirname(self.ckpt_path))
        self.cfg_path = os.path.join(self.ckpt_dir, "full_config.yaml")
                                                                                                      
        self.eval_dir = os.path.join(
            self.ckpt_dir,
            "eval_" + self.ckpt_path.split("/")[-1].split(".")[0],
            time.strftime("%Y%m%d%H%M%S"),
        )
                                       
        os.makedirs(self.eval_dir, exist_ok=True)
        print(f"eval dir: {self.eval_dir}")

        self.cfg = OmegaConf.load(self.cfg_path)
        task_cfg = self.cfg.suite.task
        self.task_name = task_cfg.tasks[0]["desktop"][0]
        print(self.task_name)
        self.task_config = SIM_TASK_CONFIGS[self.task_name]                                                               
        self.speedup = _env_bool("EVAL_SPEEDUP", True)
        self.temporal_agg = _env_bool("EVAL_TEMPORAL_AGG", True)
        self.save_video = _env_bool("EVAL_SAVE_VIDEO", False)
        self.use_nta = _env_bool("EVAL_USE_NTA", False)
        self.repeat_times = _env_int("EVAL_REPEAT_TIMES", 1)
        self.num_rollouts = _env_int("EVAL_NUM_ROLLOUTS", 50)
        self.max_timesteps = _env_int("EVAL_MAX_TIMESTEPS", 400)
        self.norm_to_minor = _env_bool("EVAL_NORM_TO_MINOR", True)
        self.save_plots = _env_bool("EVAL_SAVE_PLOTS", True)

        self.stats = {}
        self.state_dim = self.cfg["agent"]["proprio_dim"]
        self.img_key_map = {"top": "cam_global"}
        self.camera_names = ["top"]
        self.aggregator = NonlinearTemporalAgg()
        print(
            "eval config: "
            f"speedup={self.speedup}, temporal_agg={self.temporal_agg}, "
            f"use_nta={self.use_nta}, num_rollouts={self.num_rollouts}, "
            f"max_timesteps={self.max_timesteps}, save_video={self.save_video}, "
            f"save_plots={self.save_plots}"
        )
        if self.use_nta:
            print(
                "nta config: "
                f"window_range={self.aggregator.window_range}, "
                f"recency_decay={self.aggregator.recency_decay}, "
                f"max_candidates={self.aggregator.max_candidates}"
            )

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"creating agent")
        self.agent = hydra.utils.instantiate(self.cfg.agent).to(self.device)
        print(f"agent created")

        self.ckpt_names = [self.ckpt_path]
        self.img_size = tuple(self.cfg.agent.img_size)
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.resize_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        print(self.img_size)

        results = []
        for ckpt_name in self.ckpt_names:
            success_rate, avg_return = self.eval(
                ckpt_name, save_episode=self.save_video
            )
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")

    def _build_task_prompt(self):
                                                                                       
        eval_paths = self.cfg.suite.task.get("eval_data_paths", [])
        if len(eval_paths) > 0 and os.path.exists(eval_paths[0]):
            with h5py.File(eval_paths[0], "r") as f:
                if "task_emb" in f:
                    task_emb = f["task_emb"][:].astype(np.float32)
                    norm = np.linalg.norm(task_emb)
                    if norm > 1e-8:
                        task_emb = task_emb / norm
                    prompt = (
                        torch.from_numpy(task_emb)
                        .to(self.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    print(
                        f"loaded task_emb from dataset: {eval_paths[0]}, "
                        f"shape={task_emb.shape}, norm={np.linalg.norm(task_emb):.4f}"
                    )
                    return prompt

        task_emb = np.zeros((384,), dtype=np.float32)
        prompt = (
            torch.from_numpy(np.asarray(task_emb, dtype=np.float32))
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        print(
            f"fallback zero task_emb shape: {task_emb.shape}, norm={np.linalg.norm(task_emb):.4f}"
        )
        return prompt

    def load_snapshot(self, snapshots):
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f, weights_only=False)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f, weights_only=False)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload)

    def get_image(self, ts):
        all_images = {}
        for cam_name in self.camera_names:                           
            image = ts.observation["images"][cam_name]             
            if image.shape[0] != self.img_size[0] or image.shape[1] != self.img_size[1]:
                image_t = self.resize_aug(image)
            else:
                image_t = self.aug(image)
            all_images[self.img_key_map[cam_name]] = image_t.unsqueeze(0).to(self.device)
        return all_images

    def denorm_action(self, action):
        max_tensor = torch.tensor(self.stats["max"], device=action.device)
        min_tensor = torch.tensor(self.stats["min"], device=action.device)
        return (action + 1) / 2 * (max_tensor - min_tensor) + min_tensor

    def eval(self, ckpt_name, save_episode=True):
        set_seed(2000)
        
        snapshots = {}
        print(f"loading bc weight: {ckpt_name}")
        snapshots["bc"] = Path(ckpt_name)
        self.load_snapshot(snapshots)
        print(f"Loaded: {ckpt_name}")
        self.agent.cuda()
                                                                                       
        self.agent.train(False)
         
        stats_path = os.path.join(self.ckpt_dir, f"stats.hdf5")
        stats_path = Path(stats_path)
        with h5py.File(stats_path, "r") as f:
            self.stats["min"] = f["min"][:]
            self.stats["max"] = f["max"][:]

        prompt = self._build_task_prompt()
              
        env = make_sim_env(self.task_name, self.speedup)
        env_max_reward = env.task.max_reward
        print(f"env_max_reward:{env_max_reward}")

        num_queries = self.cfg.agent["num_queries"]
        if self.temporal_agg:
            query_frequency = 1
            start_idx = 0
            end_idx = num_queries
        else:
            start_idx = 0
            end_idx = num_queries
            query_frequency = num_queries - start_idx
        print(start_idx, end_idx, query_frequency)

        max_timesteps = int(self.max_timesteps)

        episode_returns = []
        highest_rewards = []
        episode_lens = []
        for rollout_id in range(self.num_rollouts):
            rollout_id += 0
            self.aggregator.reset()

                        
            if "sim_transfer_cube" in self.task_name:
                BOX_POSE[0] = sample_box_pose()                     
            elif "sim_insertion" in self.task_name:
                BOX_POSE[0] = np.concatenate(
                    sample_insertion_pose()
                )                     

            ts = env.reset()
                
            onscreen_render = False
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(
                    env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                )
                plt.ion()
            
            if self.temporal_agg:
                all_time_actions = torch.zeros(
                    [max_timesteps, max_timesteps + num_queries, self.state_dim]
                ).cuda()

            qpos_history = torch.zeros((1, max_timesteps, self.state_dim)).cuda()
            image_list = []                     
            qpos_list = []
            target_qpos_list = []
            rewards = []
            a_list = []                              
            with torch.inference_mode():
                for t in tqdm(range(max_timesteps)):
                                                              
                    if onscreen_render:
                        image = env._physics.render(
                            height=480, width=640, camera_id=onscreen_cam
                        )
                        plt_img.set_data(image)
                        plt.pause(DT)
                                                     
                    obs = ts.observation
                    image_list.append(obs["images"])

                    qpos_numpy = np.array(obs["qpos"])
                    qpos = qpos_numpy
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos
                    curr_images = self.get_image(ts)
                                 
                    if t % query_frequency == 0:
                        all_actions_list = []
                        pred_a_list = []
                        for _ in range(self.repeat_times):
                            act_kwargs = {"norm_to_minor": self.norm_to_minor}

                            all_actions, a = self.agent.act(
                                curr_images,
                                qpos,
                                prompt,
                                self.stats,
                                **act_kwargs,
                            )
                                                                                                                                   
                            all_actions = self.denorm_action(all_actions)
                            all_actions = all_actions[:, start_idx:end_idx, :]

                            all_actions_list.append(all_actions)
                            pred_a_list.append(a)

                        all_actions_list = torch.stack(
                            all_actions_list, dim=0
                        )                     
                        pred_a_list = torch.stack(pred_a_list, dim=0)            
                        all_actions = all_actions_list.mean(dim=0)
                        a = pred_a_list.mean(dim=0)
                        a_list.append(a.item())
                                      
                        action_std = (
                            all_actions_list.std(dim=0).mean().item()
                        )
                        action_mean = all_actions_list.mean().item()
                        action_cv = action_std / (
                            abs(action_mean) + 1e-8
                        )

                        left_gripper = all_actions_list[
                            :, :, :, 6
                        ]                         
                        right_gripper = all_actions_list[:, :, :, 13]
                        left_mean, left_std = (
                            left_gripper.mean().item(),
                            left_gripper.std(dim=0).mean().item(),
                        )
                        right_mean, right_std = (
                            right_gripper.mean().item(),
                            right_gripper.std(dim=0).mean().item(),
                        )

                        speed_std = pred_a_list.std(dim=0).item()
                        speed_mean = a.item()
                        speed_cv = speed_std / (
                            abs(speed_mean) + 1e-8
                        )
                                                                          
                    if self.temporal_agg:
                        action_idx = -1
                        if self.use_nta:
                                                            
                            all_actions = all_actions.squeeze(0)
                            raw_action = self.aggregator.record_and_get_current_actions(
                                all_actions, a, t
                            )
                        else:
                                                        
                            all_time_actions[[t], t : t + num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
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
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            raw_action = (actions_for_curr_step * exp_weights).sum(
                                dim=0, keepdim=True
                            )
                    else:
                        offset = t % query_frequency
                        action_idx = min(offset, all_actions.shape[1] - 1)
                        raw_action = all_actions[:, action_idx]
                                             
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = raw_action
                    target_qpos = action
                    
                    ts = env.step(target_qpos)
                 
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(ts.reward)

                    if np.array(ts.reward) == env_max_reward:
                        break

                plt.close()

            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards != None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            if episode_highest_reward == env_max_reward:
                episode_lens.append(t)
            print(
                f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}"
            )

            if save_episode:
                save_videos_with_frame_idx(
                    image_list,
                    DT,
                    video_path=os.path.join(self.eval_dir, f"rollout{rollout_id}.mp4"),
                    query_frequency=query_frequency,
                    speed_list=a_list,
                )

            if self.save_plots:
                speed_range = self.cfg.agent.new_loss_args.speed_range
                plt.figure(figsize=(10, 4))
                a_x = np.arange(len(a_list)) * query_frequency
                plt.plot(
                    a_x, a_list, color="b", linestyle="--", lw=1.5, marker="o", markersize=3
                )
                plt.xlabel("timestep")
                plt.ylabel("a")
                plt.ylim(speed_range[0], speed_range[1])
                plt.title(f"Rollout {rollout_id} - Predicted Speed (a)")
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    os.path.join(self.eval_dir, f"rollout{rollout_id}_a.png"), dpi=100
                )
                plt.close()

                n_groups = qpos_numpy.shape[-1]
                tstep = np.linspace(0, 1, len(qpos_list) - 1)
                fig, axes = plt.subplots(
                    nrows=n_groups, ncols=1, figsize=(8, 2 * n_groups), sharex=True
                )

                for n, ax in enumerate(axes):
                    ax.plot(tstep, np.array(qpos_list)[1:, n], label=f"real qpos {n}")
                    ax.plot(
                        tstep, np.array(target_qpos_list)[:-1, n], label=f"target qpos {n}"
                    )
                    ax.set_title(f"qpos {n}")
                    ax.legend()

                plt.xlabel("timestep")
                plt.ylabel("qpos")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.eval_dir, f"rollout{rollout_id}_qpos.png"), dpi=100
                )
                plt.close()

        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
        avg_return = np.mean(episode_returns)
        avg_len = float("nan")
        if len(episode_lens) > 0:
            avg_len = np.sum(episode_lens) / len(episode_lens)
        summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\nAverage length: {avg_len}\n\n"

        for r in range(env_max_reward + 1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / self.num_rollouts
            summary_str += f"Reward >= {r}: {more_or_equal_r}/{self.num_rollouts} = {more_or_equal_r_rate * 100}%\n"

        print(summary_str)
                     
        result_file_name = "result.txt"
        result_file_path = os.path.join(self.eval_dir, result_file_name)
        with open(result_file_path, "w") as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write("\n\n")
            f.write(repr(highest_rewards))
            f.write(
                f"\nspeedup: {self.speedup}\ntemporal_agg: {self.temporal_agg}\nNTA: {self.use_nta}"
                f"\nrepeat_times: {self.repeat_times}"
                f"\nNTA_window_range: {self.aggregator.window_range}"
                f"\nNTA_recency_decay: {self.aggregator.recency_decay}"
                f"\nNTA_max_candidates: {self.aggregator.max_candidates}"
            )
        return success_rate, avg_return

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an ALOHA Sim checkpoint.")
    parser.add_argument("--ckpt-path", required=True, help="Path to a training checkpoint.")
    return parser.parse_args()

def main():
    WorkspaceIL(parse_args())

if __name__ == "__main__":
    main()
