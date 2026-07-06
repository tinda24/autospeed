import os

import cv2
import numpy as np


def save_videos(video, dt, video_path=None):
    if video_path is not None:
        parent_dir = os.path.dirname(video_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for path: {video_path}")
        for image_dict in video:
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f"Saved video to: {video_path}")
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for path: {video_path}")
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]
            out.write(image)
        out.release()
        print(f"Saved video to: {video_path}")
