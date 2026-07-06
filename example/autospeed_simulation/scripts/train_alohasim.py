USED_YAML = "train_alohasim.yaml"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*Field.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*Field.*")

import h5py
import os
import sys
import csv
import time
import socket
from tqdm import tqdm
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from read_data.alohasim_eval_episode_loader import AlohasimEvalEpisodeLoader
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TZ"] = "Asia/Shanghai"

time.tzset()
import hydra
import cv2
import numpy as np
import utils.agent_utils as utils
from utils.logger_ddp import Logger, plot_losses
from utils.replay_buffer import make_expert_replay_loader
torch.backends.cudnn.benchmark = True
from omegaconf import OmegaConf

def make_optimizer(cfg,agent):
    ratio_params = [p for p in agent.ratio_head.parameters() if p.requires_grad]
    base_params = [
        p for name, p in agent.named_parameters()
        if p.requires_grad and not name.startswith("ratio_head.")
    ]

    optim_ratio_head = torch.optim.AdamW(
        ratio_params,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
    )

    optim_base = torch.optim.AdamW(
        base_params,
        lr=cfg.lr * 0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-4,
    )
    optim = {
        'ratio_head': optim_ratio_head,
        'base': optim_base,
    }
    return optim

class WorkspaceIL:
    def __init__(self, cfg, rank=0, world_size=1):
        self.work_dir = Path.cwd()
        self.rank = rank
        self.world_size = world_size
        print(f"rank={rank} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
        f"current_device={torch.cuda.current_device()} | name={torch.cuda.get_device_name(torch.cuda.current_device())}")
        if rank == 0:
            print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.cfg.dataloader.bc_dataset.stats_path = self.work_dir / "stats.hdf5"

        utils.set_seed_everywhere(cfg.seed + rank)
        self.device = torch.device(f'cuda:{rank}')

        print(f'rank {rank} create agent')
        self.agent = hydra.utils.instantiate(cfg.agent).to(self.device)
        print(f'rank {rank} agent created')

        with open(self.work_dir / "agent_config.yaml", "w") as f:
            OmegaConf.save(self.cfg.agent, f)
        with open(self.work_dir / "full_config.yaml", "w") as f:
            OmegaConf.save(self.cfg, f)

        dataset_config = dict(self.cfg.dataloader.bc_dataset)
        if world_size > 1:
            dataset_config['rank'] = rank
            dataset_config['world_size'] = world_size
        else:
            dataset_config['rank'] = 0
            dataset_config['world_size'] = 1
        dataset_iterable = hydra.utils.call(dataset_config)
        self.expert_replay_loader = make_expert_replay_loader(dataset_iterable, self.cfg.batch_size, self.cfg.num_workers)
        self.expert_replay_iter = iter(self.expert_replay_loader)

        if self.cfg.eval and self.rank == 0:
            print(f'rank {rank} create eval loader')
            self.eval_loader = AlohasimEvalEpisodeLoader(
                self.cfg.suite.task.eval_data_paths,
                stats_path=self.cfg.dataloader.bc_dataset.stats_path,
                img_size=self.cfg.dataloader.bc_dataset.img_size,
                action_overcollect_ratio=self.cfg.dataloader.bc_dataset.action_overcollect_ratio,
                num_queries=self.cfg.agent.num_queries,
                speed_range=self.cfg.agent.new_loss_args.speed_range,
            )
        else:
            self.eval_loader = None

        self.logger = Logger(
            self.work_dir,
            use_tb=True,
            rank=rank,
        )

        if world_size > 1:
            self.agent = DDP(self.agent, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        print(f'if isinstance(self.agent, DDP):{isinstance(self.agent, DDP)}')
        model_for_opt = self.agent.module if isinstance(self.agent, DDP) else self.agent
        self.optimizers = make_optimizer(cfg, model_for_opt)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.speed_message_save_path = self.work_dir / "speed_message.csv"
        self.speed_message_save_path.parent.mkdir(parents=True, exist_ok=True)

    def _maybe_barrier(self):
                                                          
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step
                 
    def train(self):
        if self.cfg.load_bc == True:
            print(f'load bc_wight from {self.cfg.bc_weight}')
            self.load_snapshot(self.cfg.bc_weight)
            print(f'successfully loaded bc_wight!')
        print(f'work_dir:{self.work_dir}')
        train_until_step = utils.Until(self.cfg.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.log_every_steps, 1)
        save_every_step = utils.Every(self.cfg.save_every_steps, 1)
        fig_every_step = utils.Every(self.cfg.fig_every_steps, 1)

        if self.world_size > 1:
            if self.rank == 0:
                print("=" * 50)
                print("All ranks ready. Starting training...")
                print("=" * 50)
            dist.barrier()

        metrics = None

        while train_until_step(self.global_step):

            agent = self.agent.module if isinstance(self.agent, DDP) else self.agent
            metrics, task_name, episode_id, sample_idx, a_selected, stage = agent.update(self.expert_replay_iter, self.global_step)

            loss = metrics['loss']
            loss_ratio = metrics['loss_ratio']
            total_loss = loss + loss_ratio

            self.optimizers['base'].zero_grad()
            if loss_ratio != 0.0:
                self.optimizers['ratio_head'].zero_grad()

            total_loss.backward()

            self.optimizers['base'].step()
            if loss_ratio != 0.0:
                self.optimizers['ratio_head'].step()

            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("loss", metrics["loss"])
                    log("step", self.global_step)

                file_exists = self.speed_message_save_path.exists()
                if self.cfg.save_speed_detail and stage != 's1':
                    with open(self.speed_message_save_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["global_step", "task_name", "episode_id", "sample_idx", "a_selected"])

                        for i in range(len(a_selected)):
                            writer.writerow([self.global_step, task_name[i], episode_id[i].item(), sample_idx[i].item(), a_selected[i].item()])

            if fig_every_step(self.global_step):
                plot_losses(self.work_dir,self.rank)
          
            if save_every_step(self.global_step):
                self._maybe_barrier()
                if self.rank == 0:
                    self.save_snapshot()
                self._maybe_barrier()

            if self.cfg.eval and self.global_step % self.cfg.eval_every_steps == 0:
                if self.rank == 0:
                    print( f"start eval at training step {self.global_step}")
                    if self.cfg.suite.task.eval_data_paths is not None:
                        self.export_eval_curve(agent, self.cfg.suite.task.eval_data_paths, self.global_step)
                    else:
                        print(f"no eval data paths specified, skip eval")
                self._maybe_barrier()

            self._global_step += 1

        if self.rank == 0:
            self.logger.finish()

    def export_eval_curve(self, agent, eval_data_paths, training_step):

        total_episodes = len(eval_data_paths)
        for idx, eval_data_path in enumerate(eval_data_paths):
            print(f"[Eval Progress] Processing episode {idx + 1}/{total_episodes}: {os.path.basename(eval_data_path)}")

            episode = self.eval_loader.sample_from_filename(eval_data_path)

            save_path = os.path.join(self.work_dir, "eval_results", f"training_step_{training_step}")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"{episode['task_name']}_episode_{episode['episode_id']}.hdf5")

            with torch.no_grad():
                max_eval_batch = 200
                num_frames = int(episode["num_frames"])
                pred_actions_parts = []
                ratio_head_output_parts = []
                update_select_speed_parts = []
                gt_action_parts = []

                for start in range(0, num_frames, max_eval_batch):
                    end = min(start + max_eval_batch, num_frames)
                    obs_chunk = {}
                    for k, v in episode["obs"].items():
                        if torch.is_tensor(v) and v.shape[0] == num_frames:
                            obs_chunk[k] = v[start:end]
                        else:
                            obs_chunk[k] = v

                    episode_chunk = {
                        "num_frames": end - start,
                        "task_emb": episode["task_emb"],
                        "obs": obs_chunk,
                        "action_chunks": episode["action_chunks"][start:end],
                        "task_name": episode["task_name"],
                        "episode_id": episode["episode_id"],
                    }

                    (pred_actions_chunk,
                    gt_action_list_chunk,
                    ratio_head_output_chunk,
                    update_select_speed_chunk) = agent.eval(
                        episode_chunk,
                        norm_stats=None,
                        t_eval_sampling_times=self.cfg.t_eval_sampling_times,
                        training_step=training_step,
                    )

                    pred_actions_parts.append(pred_actions_chunk.detach().cpu())
                    ratio_head_output_parts.append(ratio_head_output_chunk.detach().cpu())
                    update_select_speed_parts.append(update_select_speed_chunk.detach().cpu())
                    if isinstance(gt_action_list_chunk, (list, tuple)):
                        gt_action_chunk = gt_action_list_chunk[0]
                    else:
                        gt_action_chunk = gt_action_list_chunk
                    gt_action_parts.append(gt_action_chunk.detach().cpu())

                pred_actions = torch.cat(pred_actions_parts, dim=0)
                gt_action_transformed = torch.cat(gt_action_parts, dim=0)
                ratio_head_output = torch.cat(ratio_head_output_parts, dim=0)
                update_select_speed = torch.cat(update_select_speed_parts, dim=0)

            with h5py.File(save_path, "w") as f:
                f.create_dataset("pred_actions", data=pred_actions.cpu().numpy())
                f.create_dataset("gt_action_transformed", data=gt_action_transformed.cpu().numpy())
                f.create_dataset("ratio_head_output", data=ratio_head_output.cpu().numpy())
                f.create_dataset("update_select_speed", data=update_select_speed.cpu().numpy())

            print(f"[Eval Progress] Episode {idx + 1}/{total_episodes} saved to: {save_path}")
        print(f"[Eval Progress] All {total_episodes} episodes processed and saved.")

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}

        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent
        new_payload = agent.save_snapshot()
        payload.update(new_payload)

        payload['optimizers'] = {
        'base': self.optimizers['base'].state_dict(),
        'ratio_head': self.optimizers['ratio_head'].state_dict(),
        }

        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshots):
            
        snapshot_path = Path(snapshots)
        with snapshot_path.open("rb") as f:
            payload = torch.load(f, map_location=self.device, weights_only=False)

        print(f"Loading checkpoint from {snapshot_path}")

        agent_payload = {}
        for k, v in payload.items():
            if k == 'optimizers':
                pass
            elif k in self.__dict__:                      
                print(f"Restoring workspace state: {k} = {v}")
                self.__dict__[k] = v
            else:                  
                agent_payload[k] = v

        print(f"Restoring agent parameters...")
        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent
        agent.load_snapshot(agent_payload)

        if 'optimizers' in payload:
            print(f"Restoring optimizer states...")
            optimizers_dict = {
                'base': self.optimizers['base'],
                'ratio_head': self.optimizers['ratio_head'],
            }

            for opt_name, opt_state_dict in payload['optimizers'].items():
                self.optimizers[opt_name].load_state_dict(opt_state_dict)

def setup_distributed(rank, world_size):
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT')
    if master_port is None:
        raise RuntimeError("MASTER_PORT must be set before initializing distributed training.")
    if rank == 0:
        print(f"Using rendezvous endpoint {master_addr}:{master_port}")
    torch.cuda.set_device(rank)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def train_worker(rank, world_size, cfg):
    try:
        setup_distributed(rank, world_size)

        workspace = WorkspaceIL(cfg, rank=rank, world_size=world_size)
        workspace.train()

    finally:
        cleanup_distributed()

@hydra.main(config_path="../cfgs", config_name=USED_YAML, version_base="1.1")
def main(cfg):
    world_size = len(cfg.multi_gpu)
    print(f"@@@  using multi-GPU training with GPUs: {cfg.multi_gpu}")
    print(f"@@@  world_size: {world_size}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.multi_gpu))

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_ADDR"] = master_addr
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())
    print(f"@@@  MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")

    mp.spawn(
        train_worker,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()