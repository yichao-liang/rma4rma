import os
import random
import shlex
import subprocess
from typing import Callable, Optional

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch as th
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper, SuccessInfoWrapper
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from stable_baselines3.common.callbacks import CheckpointCallback


class ManiSkillRGBDVecEnvWrapper(VecEnvObservationWrapper):

    def __init__(self, env):
        assert env.obs_mode == "rgbd"
        # we simply define the single env observation space. The inherited wrapper automatically computes the batched version
        single_observation_space = ManiSkillRGBDWrapper.init_observation_space(
            env.single_observation_space
        )
        super().__init__(env, single_observation_space)

    def observation(self, observation):
        return ManiSkillRGBDWrapper.convert_observation(observation)


class ManiSkillRGBDWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = self.init_observation_space(env.observation_space)

    @staticmethod
    def init_observation_space(obs_space: spaces.Dict):
        shapes: list = []
        for cam_uid in obs_space["image"]:
            cam_space = obs_space["image"][cam_uid]
            shapes.append(cam_space["depth"].shape)
        image_shapes = np.array(shapes)
        assert np.all(image_shapes[0, :2] == image_shapes[:, :2]), image_shapes
        h, w = image_shapes[0, :2]
        c = image_shapes[:, 2].sum(0)
        rgbd_space = spaces.Box(0, np.inf, shape=(c, h, w))

        obs_space["image"] = rgbd_space
        return obs_space

    @staticmethod
    def convert_observation(observation):
        images = []
        for _, cam_obs in observation["image"].items():
            depth = cam_obs["depth"]
            if len(depth.shape) == 3:
                depth = np.transpose(depth, (2, 0, 1))
            elif len(depth.shape) == 4:
                depth = depth.permute(0, 3, 1, 2)
            else:
                raise NotImplementedError

            # SB3 does not support GPU tensors; transfer to CPU.
            if isinstance(depth, th.Tensor):
                depth = depth.to(device="cpu", non_blocking=True)

            images.append(depth)

        observation["image"] = np.concatenate(images, axis=-1)
        return observation

    def observation(self, observation):
        return self.convert_observation(observation)


def tprint(*args):
    """Temporarily prints things on the screen."""
    print("\r", end="")
    print(*args, end="")


def linear_schedule(
    initial_value: float,
    final_value: float,
    init_step: int = 0,
    end_step: int = int(2e7),
    total_steps: Optional[int] = None,
) -> Callable[[float], float]:
    """Linear learning rate schedule. anneal_percent goes from 0 (beginning) to 1 (end).
    :param initial_value: Initial learning rate. :return: schedule that computes current
    learning rate depending on remaining progress.

    when initial_value = 1, final_value = 2, the diff is -1
        when percent = 10%, the current_value = 1 - 0.1 * (-1) =
        when percent = 99%, the current_value = 1 - 0.99 * (-1) = 1.99
    """

    def func(progress_remaining: Optional[float] = None, elapsed_steps=None) -> float:
        """Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if elapsed_steps is None:
            assert total_steps is not None and progress_remaining is not None
            elapsed_steps = total_steps * (1 - progress_remaining)
        if elapsed_steps < init_step:
            return initial_value
        anneal_percent = min(elapsed_steps / end_step, 1.0)
        return initial_value - anneal_percent * (initial_value - final_value)

    return func


class LatestCheckpointCallback(CheckpointCallback):

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_" or "vecnormalize_"
            for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"latest.{extension}")


def get_task_id(task_name: str) -> np.ndarray:
    """Get the task_id from task_name."""
    name_to_id = {
        "PickCube": 0,
        "PickSingleYCB": 0,
        "StackCube": 1,
    }
    return np.array([name_to_id[task_name]])


def get_object_id(
    task_name: str,
    model_id: Optional[str] = None,
    object_list: Optional[list] = None,
) -> np.ndarray:
    """Map a (task, model) pair to an integer object id used for embedding lookup.

    For tasks with a single fixed object (PickCube, StackCube, PegInsertion), a
    constant is returned. For multi-object tasks (TurnFaucet, PickSingleYCB,
    PickSingleEGAD), the id is the model's index in ``object_list`` (offset by 2
    so 0/1 stay reserved for the fixed-object tasks).
    """
    if task_name in ["PickCube", "StackCube"]:
        return np.array([1])
    if task_name == "PegInsertion":
        return np.array([0])
    if task_name == "TurnFaucet" and object_list is None:
        return np.array([0])
    if task_name in ["TurnFaucet", "PickSingleYCB", "PickSingleEGAD"]:
        assert model_id is not None and object_list is not None
        return np.array([object_list.index(model_id) + 2])
    raise NotImplementedError(f"Unknown task: {task_name}")


class RecordEpisodeRandInfo(RecordEpisode):
    """RecordEpisode subclass that skips flushing empty transitions."""

    def flush_video(self, suffix="", verbose=False, ignore_empty_transition=False):
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return
        super().flush_video(suffix, verbose, ignore_empty_transition)


# define an SB3 style make_env function for evaluation
def make_env(
    env_id: str,
    obs_mode: str = "state",
    control_mode: str = "pd_ee_delta_pose",
    reward_mode: str = "normalized_dense",
    max_episode_steps: Optional[int] = None,
    record_dir: Optional[str] = None,
    **kwargs,
):

    def _init() -> gym.Env:
        # Import envs here so they are registered with gym inside subprocesses.
        import mani_skill2.envs  # noqa: F401

        env = gym.make(
            env_id,
            obs_mode=obs_mode,
            reward_mode=reward_mode,
            control_mode=control_mode,
            renderer_kwargs={"offscreen_only": True},
            render_mode="cameras",
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
        # For training we treat tasks as continuous (no truncation signal).
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env)
        if obs_mode == "rgbd":
            env = ManiSkillRGBDWrapper(env)

        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisodeRandInfo(
                env,
                record_dir,
                info_on_video=False,
                save_trajectory=False,
            )
        return env

    return _init


def git_diff_config(name):
    cmd = f"git diff --unified=0 {name}"
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    return seed


class AverageScalarMeter(object):

    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        """Compute running average over the most recent ``window_size`` values.

        Before the buffer is full, averages over everything seen so far.
        """
        size = values.size()[0]
        if size == 0:
            return
        new_mean = th.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
