
import csv
import datetime
from collections import defaultdict
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
import wandb

BC_TRAIN_FORMAT = [
    ("step", "S", "int"),
    ("loss", "L", "float"),
    ("loss_ratio", "L_r", "float"),
    ("total_time", "T", "time"),
    ("rank", "Rank", "int"),
]
BC_EVAL_FORMAT = [
    ("frame", "F", "int"),
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("imitation_reward", "R_i", "float"),
    ("total_time", "T", "time"),
    ("rank", "Rank", "int"),
]
BEST_PARAMS_FORMAT = [
    ("step", "S", "int"),
]
for i in range(10):
    BEST_PARAMS_FORMAT.append((f"best_a_{i}", f"best_a_{i}", "float"))
for i in range(10):
    BEST_PARAMS_FORMAT.append((f"best_b_{i}", f"best_b_{i}", "float"))
BEST_PARAMS_FORMAT.extend([
    ("total_time", "T", "time"),
    ("rank", "Rank", "int"),
])
SSL_TRAIN_FORMAT = [
    ("step", "S", "int"),
    ("loss", "L", "float"),
    ("total_time", "T", "time"),
    ("rank", "Rank", "int"),
]
SSL_EVAL_FORMAT = [
    ("epoch", "E", "int"),
    ("step", "S", "int"),
    ("loss", "E", "float"),
    ("total_time", "T", "time"),
    ("rank", "Rank", "int"),
]

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating,rank):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None
        self._rank = rank
    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train_vq"):
                key = key[len("train_vq") + 1 :]
            elif key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data["rank"] = self._rank
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        compare_key = None
        for key in ["step", "frame", "episode"]:
            if key in data:
                compare_key = key
                break
        if compare_key is None:
            return
        
        with self._csv_file_name.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if compare_key not in row:
                    continue
                if float(row[compare_key]) >= data[compare_key]:
                    break
                rows.append(row)
        with self._csv_file_name.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(data.keys()), restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open("a")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=sorted(data.keys()), restval=0.0
            )
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        prefix = colored(
            prefix, "yellow" if prefix in ["train", "train_vq"] else "green"
        )
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data["frame"] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb, rank, mode='bc', use_wandb=False, wandb_config=None):
        """
        mode: bc, vqvae
        """
        self._log_dir = log_dir
        self._rank = rank
        print(f'log_dir{log_dir}')
        if mode == "bc":
            self._train_mg = MetersGroup(
                log_dir / f"train_{rank}.csv", formating=BC_TRAIN_FORMAT,rank=rank
            )
            self._best_params_mg = MetersGroup(
                log_dir / f"best_params_{rank}.csv", formating=BEST_PARAMS_FORMAT,rank=rank
            )
            self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=BC_EVAL_FORMAT,rank=rank)
        elif mode == "vqvae":
            self._train_mg = MetersGroup(
                log_dir / "train.csv", formating=SSL_TRAIN_FORMAT,rank=rank
            )
            self._train_vq_mg = MetersGroup(
                log_dir / "train_vq.csv", formating=SSL_TRAIN_FORMAT,rank=rank
            )
            self._eval_mg = MetersGroup(log_dir / "eval.csv", formating=SSL_EVAL_FORMAT,rank=rank)

        if use_tb:
            self._sw = SummaryWriter(str(log_dir / "tb"))
        else:
            self._sw = None
        
        # Initialize wandb only on rank 0
        if use_wandb and rank == 0:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get('project', 'autospeed'),
                entity=wandb_config.get('entity', None),
                name=wandb_config.get('name', None),
                config=wandb_config.get('config', {}),
                dir=str(log_dir),
                mode='offline'
            )
            self._wandb_enabled = True
            print(f"wandb initialized with project: {wandb_config.get('project', 'autospeed')}")
        else:
            self._wandb_enabled = False

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)
    
    def _try_wandb_log(self, key, value, step):
        if self._wandb_enabled:
            wandb.log({key: value}, step=step)

    def log(self, key, value, step):
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        self._try_wandb_log(key, value, step)
        if key.startswith("train_vq"):
            mg = self._train_vq_mg
        else:
            mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f"{ty}/{key}", value, step)

    def log_best_params(self, metrics, step):
        for key, value in metrics.items():
            if type(value) == torch.Tensor:
                value_np = value.detach().cpu().numpy()
                
                if value_np.ndim == 1:
                    mean_val = float(value_np.mean())
                    std_val = float(value_np.std())
                    min_val = float(value_np.min())
                    max_val = float(value_np.max())
                    
                    self._best_params_mg.log(f"train/{key}_mean", mean_val)
                    self._best_params_mg.log(f"train/{key}_std", std_val)
                    self._best_params_mg.log(f"train/{key}_min", min_val)
                    self._best_params_mg.log(f"train/{key}_max", max_val)
                    
                    if self._wandb_enabled:
                        wandb.log({
                            f"best_params/{key}_mean": mean_val,
                            f"best_params/{key}_std": std_val,
                            f"best_params/{key}_min": min_val,
                            f"best_params/{key}_max": max_val,
                        }, step=step)
                    
                    for i in range(min(10, len(value_np))):
                        val = float(value_np[i])
                        self._best_params_mg.log(f"train/{key}_{i}", val)
                        if self._wandb_enabled:
                            wandb.log({f"best_params/{key}_{i}": val}, step=step)
                else:
                    value = value.item()
                    self._best_params_mg.log(f"train/{key}", value)
                    if self._wandb_enabled:
                        wandb.log({f"best_params/{key}": value}, step=step)
            else:
                self._best_params_mg.log(f"train/{key}", value)
                if self._wandb_enabled:
                    wandb.log({f"best_params/{key}": value}, step=step)

    def dump(self, step, ty=None):
        if ty is None or ty == "eval":
            self._eval_mg.dump(step, "eval")
        if ty is None or ty == "train":
            self._train_mg.dump(step, "train")
        if ty is None or ty == "train_vq":
            self._train_vq_mg.dump(step, "train_vq")
        if ty == "best_params":
            self._best_params_mg.dump(step, "best_params")

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)
    
    def finish(self):
        """Finish wandb run"""
        if self._wandb_enabled:
            wandb.finish()
            print("wandb run finished")


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f"{self._ty}/{key}", value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)






def plot_losses(work_dir: str, rank: int = None, start_frame: int = None, end_frame: int = None):
    # print(f'work_dir: {work_dir}')
    if rank is not None:
        csv_path = os.path.join(work_dir, f"train_{rank}.csv")
    else:
        csv_path = os.path.join(work_dir, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter by frame range if specified
    if start_frame is not None:
        df = df[df["frame"] >= start_frame]
    if end_frame is not None:
        df = df[df["frame"] <= end_frame]

    plt.figure(figsize=(8, 5))
    plt.plot(df["frame"], df["loss"], label="loss")
    if "img_loss" in df.columns:
        plt.plot(df["frame"], df["img_loss"], label="img_loss")

    plt.xlabel("frame")
    plt.ylabel("loss")
    if "img_loss" in df.columns:
        plt.title("Actor Loss & Img Loss vs Frame")
    else:
        plt.title("Actor Loss vs Frame")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    imgs_dir = os.path.join(work_dir, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)
    if rank is not None:
        out_path = os.path.join(imgs_dir, f"loss_fig_{rank}.png")
    else:
        out_path = os.path.join(imgs_dir, "loss_fig.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()