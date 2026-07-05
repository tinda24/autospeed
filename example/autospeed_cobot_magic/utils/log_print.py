import csv
from pathlib import Path


def log_print_eval(*args, **kwargs):
    msg = " ".join(str(arg) for arg in args)
    print(msg, **kwargs)
    with open(Path.cwd() / "eval_info.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_print_params(*args, **kwargs):
    msg = " ".join(str(arg) for arg in args)
    print(msg, **kwargs)
    with open(Path.cwd() / "parameter_info.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_print_imle(*args, **kwargs):
    msg = " ".join(str(arg) for arg in args)
    with open(Path.cwd() / "imle_train_info.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_print_env_action_detailed(
    env_idx,
    frame_idx,
    action_query_idx,
    loss,
    global_step,
    filename="base_env_action_detail.csv",
):
    file_path = Path.cwd() / filename
    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["global_step", "env_idx", "frame_idx", "action_query_idx", "loss"])
        writer.writerow([global_step, env_idx, frame_idx, action_query_idx, loss])


def log_to_csv_episode_frame_detailed(
    frame_idx,
    loss,
    global_step,
    filename="episode_frame_loss.csv",
):
    file_path = Path.cwd() / filename
    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["global_step", "frame_idx", "loss"])
        writer.writerow([global_step, frame_idx, loss])


def log_to_csv_imgloss_detailed(
    frame_idx,
    pixel_keys,
    img_loss,
    global_step,
    filename="img_loss_by_view.csv",
):
    file_path = Path.cwd() / filename
    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["global_step", "frame_idx", "pixel_keys", "img_loss"])
        for i, key in enumerate(pixel_keys):
            writer.writerow([global_step, frame_idx, key, img_loss[:, i].mean()])
