from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

import gym
from gym import Wrapper, spaces
from gym.wrappers import FrameStack

import dm_env
import numpy as np
from dm_env import StepType, specs, TimeStep

import os
import cv2
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R

_SENTENCE_MODEL_CACHE_DIR = (
    Path(__file__).resolve().parent.parent
    / "lang_weights"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
)
_sentence_encoder = None

def _encode_task(text):
    global _sentence_encoder
    refs_path = _SENTENCE_MODEL_CACHE_DIR / "refs" / "main"
    if refs_path.exists():
        if _sentence_encoder is None:
            from sentence_transformers import SentenceTransformer

            snapshot = _SENTENCE_MODEL_CACHE_DIR / "snapshots" / refs_path.read_text().strip()
            _sentence_encoder = SentenceTransformer(str(snapshot))
        return _sentence_encoder.encode(text)
    return np.zeros((384,), dtype=np.float32)

def _set_use_delta(env, use_delta: bool):
    for robot in env.env.robots:
        robot.controller.use_delta = bool(use_delta)


def _get_zero_action(env):
    try:
        action_spec = env.env.action_spec
        shape = action_spec[0].shape
    except Exception:
        shape = (7,)
    a = np.zeros(shape, dtype=np.float32)
    if a.size > 0:
        a[-1] = -1.0
    return a

def _rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    u2 = a2 - np.dot(b1, a2) * b1
    b2 = u2 / max(np.linalg.norm(u2), 1e-12)
    b3 = np.cross(b1, b2)
    mat = np.stack([b1, b2, b3], axis=-1)
    return R.from_matrix(mat).as_rotvec().astype(np.float32)

def _maybe_convert_action_to_env_dim(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == env_action_dim:
        return action

    if env_action_dim == 7 and action.shape[0] == 10:
        xyz = action[0:3]
        rotvec = _rot6d_to_rotvec(action[3:9])
        gripper = action[9:10]
        return np.concatenate([xyz, rotvec, gripper], axis=0).astype(np.float32)

    raise ValueError(
        f"Action dim mismatch: got {action.shape[0]}, expected {env_action_dim}. "
        "Supported conversion: 10D(xyz+rot6d+gripper) -> 7D(xyz+axis-angle+gripper)."
    )

class LiberoControllerModeWrapper:
    def __init__(
        self,
        env,
        use_delta: bool = True,
    ):
        self._env = env
        self._use_delta = bool(use_delta)

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        _burn_in_steps = 0
        if _burn_in_steps > 0:
            _set_use_delta(self._env, True)
            zero_action = _get_zero_action(self._env)
            for _ in range(_burn_in_steps):
                obs, reward, done, info = self._env.step(zero_action)
        _set_use_delta(self._env, self._use_delta)
        return obs

    def step(self, action):
        return self._env.step(action)

    def __getattr__(self, name):
        return getattr(self._env, name)

class LiberoGainTuningWrapper:
    def __init__(
        self,
        env,
        arm_kp_scale: float = 1.0,
        arm_kd_scale: float = 1.0,
        gripper_gain_scale: float = 1.0,
    ):
        self._env = env
        self._arm_kp_scale = float(arm_kp_scale)
        self._arm_kd_scale = float(arm_kd_scale)
        self._gripper_gain_scale = float(gripper_gain_scale)

        self._base_kp = None
        self._base_kd = None
        self._orig_actuator_gainprm = None
        self._orig_actuator_biasprm = None

    def _apply_arm_gains(self):
        if self._arm_kp_scale == 1.0 and self._arm_kd_scale == 1.0:
            return
        controller = self._env.env.robots[0].controller
        if self._base_kp is None:
            self._base_kp = np.array(controller.kp, copy=True)
        if self._base_kd is None:
            self._base_kd = np.array(controller.kd, copy=True)
        controller.kp = self._base_kp * self._arm_kp_scale
        controller.kd = self._base_kd * self._arm_kd_scale

    def _apply_gripper_actuator_gains(self):
        model = self._env.env.sim.model
        indices = [7, 8]

        if self._orig_actuator_gainprm is None:
            self._orig_actuator_gainprm = np.array(model.actuator_gainprm, copy=True)
        if self._orig_actuator_biasprm is None:
            self._orig_actuator_biasprm = np.array(model.actuator_biasprm, copy=True)

        model.actuator_gainprm[indices, :] = (
            self._orig_actuator_gainprm[indices, :] * self._gripper_gain_scale
        )
        model.actuator_biasprm[indices, :] = (
            self._orig_actuator_biasprm[indices, :] * self._gripper_gain_scale
        )

    def _apply(self):
        self._apply_arm_gains()
        self._apply_gripper_actuator_gains()

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        self._apply()
        return obs

    def step(self, action):
        return self._env.step(action)

    def __getattr__(self, name):
        return getattr(self._env, name)

class RGBArrayAsObservationWrapper(dm_env.Environment):
    def __init__(
        self, env, width=84, height=84, max_episode_len=300, max_state_dim=100
    ):
        self._env = env
        self._width = width
        self._height = height
        self._max_episode_len = max_episode_len
        obs = self._env.reset()
        state = self._env.get_sim_state()
        self._max_state_dim = max(max_state_dim, state.shape[0])
        dummy_obs = obs["agentview_image"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype
        )

                  
        self.task_emb = _encode_task(self._env.language_instruction)

                     
        action_spec = self._env.env.action_spec
        self._action_spec = specs.BoundedArray(
            action_spec[0].shape, np.float32, action_spec[0], action_spec[1], "action"
        )
                          
        robot_state = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
        self._obs_spec = {}
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=dummy_obs.shape, dtype=np.uint8, minimum=0, maximum=255, name="pixels"
        )
        self._obs_spec["pixels_egocentric"] = specs.BoundedArray(
            shape=dummy_obs.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels_egocentric",
        )
        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=robot_state.shape,
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="proprioceptive",
        )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        self.render_image = None

    def reset(self, **kwargs):
        self._step = 0
        obs = self._env.reset(**kwargs)
        self.render_image = obs["agentview_image"][::-1, :]

        observation = {}
        observation["pixels"] = obs["agentview_image"][::-1, :]
        observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        observation["proprioceptive"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
                   
        observation["features"] = np.zeros(self._max_state_dim)
        state = self._env.get_sim_state()
        if state.shape[0] > self._max_state_dim:
                                                                             
            self._max_state_dim = state.shape[0]
            observation["features"] = np.zeros(self._max_state_dim)
        observation["features"][: state.shape[0]] = state
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = False
        return observation

    def step(self, action):
        self._step += 1
        obs, reward, done, info = self._env.step(action)
        self.render_image = obs["agentview_image"][::-1, :]

        observation = {}
        observation["pixels"] = obs["agentview_image"][::-1, :]
        observation["pixels_egocentric"] = obs["robot0_eye_in_hand_image"][::-1, :]
        observation["proprioceptive"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
                   
        observation["features"] = np.zeros(self._max_state_dim)
        state = self._env.get_sim_state()
        if state.shape[0] > self._max_state_dim:
            self._max_state_dim = state.shape[0]
            observation["features"] = np.zeros(self._max_state_dim)
        observation["features"][: state.shape[0]] = state
        observation["task_emb"] = self.task_emb
        observation["goal_achieved"] = done
        return observation, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return cv2.resize(self.render_image, (width, height))

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)

class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._frames_egocentric = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()["pixels"]

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec["features"] = self._env.observation_spec()["features"]
        self._obs_spec["proprioceptive"] = self._env.observation_spec()[
            "proprioceptive"
        ]
        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels",
        )
        self._obs_spec["pixels_egocentric"] = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels_egocentric",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        assert len(self._frames_egocentric) == self._num_frames
        obs = {}
        obs["features"] = time_step.observation["features"]
        obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        obs["pixels_egocentric"] = np.concatenate(list(self._frames_egocentric), axis=0)
        obs["proprioceptive"] = time_step.observation["proprioceptive"]
        obs["task_emb"] = time_step.observation["task_emb"]
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation["pixels"]
        pixels_egocentric = time_step.observation["pixels_egocentric"]

                          
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        if len(pixels_egocentric.shape) == 4:
            pixels_egocentric = pixels_egocentric[0]
        return (
            pixels.transpose(2, 0, 1).copy(),
            pixels_egocentric.transpose(2, 0, 1).copy(),
        )

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        pixels, pixels_egocentric = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels, pixels_egocentric = self._extract_pixels(time_step)
        self._frames.append(pixels)
        self._frames_egocentric.append(pixels_egocentric)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

                     
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        env_action_dim = int(np.prod(self._env.action_spec().shape))
        action = _maybe_convert_action_to_env_dim(action, env_action_dim)
        action = action.astype(self._env.action_spec().dtype)
                                         
        observation, reward, done, info = self._env.step(action)
        step_type = (
            StepType.LAST
            if (
                self._env._step == self._env._max_episode_len
                or observation["goal_achieved"]
            )
            else StepType.MID
        )
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(
    suite,
    scenes,
    tasks,
    frame_stack,
    action_repeat,
    seed,
    height,
    width,
    max_episode_len,
    max_state_dim,
    eval,
    use_delta=True,
    gripper_gain_scale: float = 1.0,
    controller_kp_scale: float = 1.0,
    controller_kd_scale: float = 1.0,
):
                                                          
    tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}
    envs = []
    task_descriptions = []
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    for scene in scenes:
        for task_name in tasks[scene]:
            if task_name in task_suite.get_task_names():
                                                     
                task_id = task_suite.get_task_names().index(task_name)
                                    
                task = task_suite.get_task(task_id)
                task_name = task.name
                task_descriptions.append(task.language)
                task_bddl_file = os.path.join(
                    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
                )
                env_args = {
                    "bddl_file_name": task_bddl_file,
                    "camera_heights": 128,
                    "camera_widths": 128,
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(seed)
                env = LiberoControllerModeWrapper(env, use_delta=use_delta)
                                                                   
                env = LiberoGainTuningWrapper(env, arm_kp_scale=controller_kp_scale, arm_kd_scale=controller_kd_scale, gripper_gain_scale=gripper_gain_scale)
                print(f"Initialized environment: {task_name}")

                                
                env = RGBArrayAsObservationWrapper(
                    env,
                    height=height,
                    width=width,
                    max_episode_len=max_episode_len,
                    max_state_dim=max_state_dim,
                )
                env = ActionDTypeWrapper(env, np.float32)
                env = ActionRepeatWrapper(env, action_repeat)
                env = FrameStackWrapper(env, frame_stack)
                env = ExtendedTimeStepWrapper(env)

                envs.append(env)
            else:
                for task_id in range(task_suite.get_num_tasks()):
                    task = task_suite.get_task(task_id)
                    task_name = task.name
                    task_descriptions.append(task.language)
                    task_bddl_file = os.path.join(
                        get_libero_path("bddl_files"),
                        task.problem_folder,
                        task.bddl_file,
                    )
                    env_args = {
                        "bddl_file_name": task_bddl_file,
                        "camera_heights": 128,
                        "camera_widths": 128,
                    }
                    env = OffScreenRenderEnv(**env_args)
                    env.seed(seed)
                    env = LiberoControllerModeWrapper(env, use_delta=use_delta)                                                 
                    env = LiberoGainTuningWrapper(env, arm_kp_scale=controller_kp_scale, arm_kd_scale=controller_kd_scale, gripper_gain_scale=gripper_gain_scale)
              
                    env = RGBArrayAsObservationWrapper(
                        env,
                        height=height,
                        width=width,
                        max_episode_len=max_episode_len,
                        max_state_dim=max_state_dim,
                    )
                    env = ActionDTypeWrapper(env, np.float32)
                    env = ActionRepeatWrapper(env, action_repeat)
                    env = FrameStackWrapper(env, frame_stack)
                    env = ExtendedTimeStepWrapper(env)

                    envs.append(env)

                    if not eval:
                        break

            if not eval:
                break
        if not eval:
            break

    return envs, task_descriptions
