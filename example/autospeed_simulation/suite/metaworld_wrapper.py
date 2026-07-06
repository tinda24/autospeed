
import sys
import numpy as np
import collections
import os
import re
from termcolor import cprint
import torch
import cv2

metaworld_root = os.environ.get("METAWORLD_ROOT")
if metaworld_root and metaworld_root not in sys.path:
    sys.path.append(metaworld_root)
try:
    import metaworld
    import mujoco
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
except ImportError:
    print("MetaWorld not found/installed correctly.")
                                                                               
_TASK_DESC_TO_V2_ENV = {
    "pick-up-a-nut-and-place-it-onto-a-peg": "assembly-v2",
    "dunk-the-basketball-into-the-basket": "basketball-v2",
    "grasp-the-puck-from-one-bin-and-place-it-into-another-bin": "bin-picking-v2",
    "grasp-the-cover-and-close-the-box-with-it": "box-close-v2",
    "press-a-button-from-the-top": "button-press-topdown-v2",
    "bypass-a-wall-and-press-a-button-from-the-top": "button-press-topdown-wall-v2",
    "press-a-button": "button-press-v2",
    "bypass-a-wall-and-press-a-button": "button-press-wall-v2",
    "push-a-button-on-the-coffee-machine": "coffee-button-v2",
    "pull-a-mug-from-a-coffee-machine": "coffee-pull-v2",
    "push-a-mug-under-a-coffee-machine": "coffee-push-v2",
    "rotate-a-dial-180-degrees": "dial-turn-v2",
    "pick-a-nut-out-of-a-peg": "disassemble-v2",
    "close-a-door-with-a-revolving-joint": "door-close-v2",
    "lock-the-door-by-rotating-the-lock-clockwise": "door-lock-v2",
    "open-a-door-with-a-revolving-joint": "door-open-v2",
    "unlock-the-door-by-rotating-the-lock-counter-clockwise": "door-unlock-v2",
    "insert-the-gripper-into-a-hole": "hand-insert-v2",
    "push-and-close-a-drawer": "drawer-close-v2",
    "open-a-drawer": "drawer-open-v2",
    "rotate-the-faucet-counter-clockwise": "faucet-open-v2",
    "rotate-the-faucet-clockwise": "faucet-close-v2",
    "hammer-a-screw-on-the-wall": "hammer-v2",
    "press-a-handle-down-sideways": "handle-press-side-v2",
    "press-a-handle-down": "handle-press-v2",
    "pull-a-handle-up-sideways": "handle-pull-side-v2",
    "pull-a-handle-up": "handle-pull-v2",
    "pull-a-lever-down-90-degrees": "lever-pull-v2",
    "pick-a-puck-bypass-a-wall-and-place-the-puck": "pick-place-wall-v2",
    "pick-up-a-puck-from-a-hole": "pick-out-of-hole-v2",
    "pick-and-place-a-puck-to-a-goal": "pick-place-v2",
    "slide-a-plate-into-a-cabinet": "plate-slide-v2",
    "slide-a-plate-into-a-cabinet-sideways": "plate-slide-side-v2",
    "get-a-plate-from-the-cabinet": "plate-slide-back-v2",
    "get-a-plate-from-the-cabinet-sideways": "plate-slide-back-side-v2",
    "insert-a-peg-sideways": "peg-insert-side-v2",
    "unplug-a-peg-sideways": "peg-unplug-side-v2",
    "kick-a-soccer-into-the-goal": "soccer-v2",
    "grasp-a-stick-and-push-a-box-using-the-stick": "stick-push-v2",
    "grasp-a-stick-and-pull-a-box-with-the-stick": "stick-pull-v2",
    "push-the-puck-to-a-goal": "push-v2",
    "bypass-a-wall-and-push-a-puck-to-a-goal": "push-wall-v2",
    "reach-a-goal-position": "reach-v2",
    "bypass-a-wall-and-reach-a-goal": "reach-wall-v2",
    "pick-and-place-a-puck-onto-a-shelf": "shelf-place-v2",
    "sweep-a-puck-into-a-hole": "sweep-into-v2",
    "sweep-a-puck-off-the-table": "sweep-v2",
    "push-and-open-a-window": "window-open-v2",
    "push-and-close-a-window": "window-close-v2",
}

def _valid_v2_env_names():
    return set(getattr(metaworld.ML1, "ENV_NAMES", []))


def _strip_dataset_task_prefix(task_name: str) -> str:
    name = task_name.strip().lower()
    match = re.match(r"^task_\d+_(.+)$", name)
    return match.group(1) if match else name


def resolve_metaworld_task_name(task_name: str) -> str:
    if task_name is None:
        raise ValueError("task_name cannot be None")

    raw_name = str(task_name).strip()
    if raw_name == "":
        raise ValueError("task_name cannot be empty")

    valid_v2 = _valid_v2_env_names()
    lower_name = raw_name.lower()

    if lower_name in valid_v2:
        return lower_name

    if lower_name.endswith("-goal-observable"):
        base_name = lower_name[: -len("-goal-observable")]
        if base_name in valid_v2:
            return base_name

    desc_name = _strip_dataset_task_prefix(lower_name)
    mapped = _TASK_DESC_TO_V2_ENV.get(desc_name, None)
    if mapped is not None:
        return mapped

    if f"{desc_name}-v2" in valid_v2:
        return f"{desc_name}-v2"

    raise ValueError(
        f"Unknown MetaWorld task name: '{task_name}'. "
        "Use a valid V2 env name like 'assembly-v2', or dataset task folder "
        "name like 'task_000_pick-up-a-nut-and-place-it-onto-a-peg'."
    )


class MetaWorldAbsoluteWrapper(object):
    _CORNER2_CAM_POS = np.array([0.75, 0.075, 0.7], dtype=np.float64)

    def __init__(
        self,
        env,
        task_name=None,
        camera_name="corner2",
    ):
        self._env = env
        self.task_name = task_name
        self.camera_name = camera_name
        self._camera_id = None
      
        self.action_space = env.action_space
        self.observation_space = env.observation_space                             
                 
        self.action_scale = getattr(env, 'action_scale', 1.0/100)
        self.mocap_low = getattr(env, 'mocap_low', None)
        self.mocap_high = getattr(env, 'mocap_high', None)
                      
        self.task_emb = None
        self._last_state_obs = None

    def reset(self):
        obs, info = self._env.reset()
        return self._get_obs(obs, info)

    def step(self, action_abs):
        if self._last_state_obs is not None and self._last_state_obs.shape[0] >= 3:
            curr_pos = self._last_state_obs[:3]
        else:
            curr_pos = self._env.data.mocap_pos.flat[:3]
        target_pos = action_abs[:3]
        delta = (target_pos - curr_pos) / self.action_scale
        action_rel = np.clip(delta, -1, 1)
        gripper = action_abs[3]
        full_action_rel = np.concatenate([action_rel, [gripper]])

        obs, reward, done, truncate, info = self._env.step(full_action_rel)

        obs_dict = self._get_obs(obs, info)
        return obs_dict, reward, done, info

    def _get_obs(self, obs, info):                                 
        try:
            self._set_render_camera()
            img = self._env.render()
        except Exception:
            img = np.zeros((480, 480, 3), dtype=np.uint8)

        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._last_state_obs = obs_arr.copy()

        if obs_arr.shape[0] >= 9:
            proprio = obs_arr[:9].copy()
        else:
            proprio = self._env.data.qpos.flat[:9].copy()

                      
        if self.task_emb is None:
             self.task_emb = self._get_task_emb()

        return {
            "cam_global": img,
            "proprioceptive": proprio,
            "task_emb": self.task_emb,
            "goal_achieved": info.get('success', False)
        }

    def _get_task_emb(self):
                                        
        try:
             from sentence_transformers import SentenceTransformer
             model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
             description = self.task_name.replace('-v2', '').replace('-', ' ') if self.task_name else "task"
             emb = model.encode([description])[0]
             return emb
        except Exception as e:
             print(f"Warning: Could not encode task: {e}")
             return np.zeros(384)

    def _set_render_camera(self):
        if not self.camera_name:
            return
        renderer = getattr(self._env, "mujoco_renderer", None)
        model = getattr(self._env, "model", None)
        if renderer is None or model is None:
            return
        if self._camera_id is None:
            cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name))
            if cam_id < 0:
                return
            self._camera_id = cam_id
            if self.camera_name == "corner2":
                target = self._CORNER2_CAM_POS.astype(model.cam_pos.dtype, copy=False)
                model.cam_pos[self._camera_id] = target
                try:
                    mujoco.mj_forward(model, self._env.data)
                except Exception:
                    pass
        renderer.camera_id = self._camera_id

def make(
    task_name,
    seed=0,
    render_mode='rgb_array',
    camera_name="corner2",
    **kwargs,
):
    resolved_task_name = resolve_metaworld_task_name(task_name)
                     
    goal_observable_env_id = f"{resolved_task_name}-goal-observable"
    goal_envs = getattr(metaworld.envs, "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE", {})
    if goal_observable_env_id in goal_envs:
        env = goal_envs[goal_observable_env_id]()
        env._freeze_rand_vec = False
        env._partially_observable = False
        env.render_mode = render_mode
    else:
        ml1 = metaworld.ML1(resolved_task_name)
        env = ml1.train_classes[resolved_task_name](render_mode=render_mode)
        task = [t for t in ml1.train_tasks if t.env_name == resolved_task_name][0]
        env.set_task(task)

    env.seed(seed)

    wrapper = MetaWorldAbsoluteWrapper(
        env,
        task_name=resolved_task_name,
        camera_name=camera_name,
    )
    return wrapper
