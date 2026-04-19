import os
import time

import gymnasium.spaces as spaces
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import get_device, obs_as_tensor
from torch.utils.tensorboard import SummaryWriter

from rma4rma.algo.misc import AverageScalarMeter, tprint


class ProprioAdapt:
    """Stage-2 adapter training loop.

    Freezes the base policy and environment encoder, and trains the adapter
    module via L² regression between the predicted and ground-truth
    environment embeddings.
    """

    def __init__(
        self,
        model: PPO,
        env: GymEnv,
        writer: SummaryWriter,
        save_dir: str,
    ):
        self.model = model
        self.num_timesteps = 0
        self.device = get_device("auto")
        self.nn_dir = save_dir

        self.policy = model.policy
        self.env = env
        self.logger = writer

        self.action_space = env.action_space
        self.step_reward = th.zeros(env.num_envs, dtype=th.float32)
        self.step_length = th.zeros(env.num_envs, dtype=th.float32)
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -np.inf
        self.best_succ_rate = -np.inf
        # ---- Optim ----
        adapt_params = []
        for name, p in self.policy.named_parameters():
            if "adapt_tconv" in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        self.optim = th.optim.Adam(adapt_params, lr=1e-4)

    def get_env(self):
        return None

    def learn(self):
        """Training the adaptation module."""
        if hasattr(self.model, "adaptation_steps"):
            n_steps = self.model.adaptation_steps
            self.best_succ_rate = self.model.best_succ_rate
        else:
            n_steps = 0
            self.model.best_succ_rate = 0
        self.succ_rate = 0

        _t = time.time()
        _last_t = time.time()
        self._last_obs = self.env.reset()
        n_envs = self.env.num_envs
        assert self._last_obs is not None, "No previous observation was provided"
        while n_steps <= 1e6:
            # Convert to pytorch tensor
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

            for key, tensor in obs_tensor.items():
                obs_tensor[key] = tensor.detach()
            actions, _, _, e, e_gt = self.policy(obs_tensor, adapt_trn=True)

            loss = ((e - e_gt.detach()) ** 2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Rescale and perform action
            actions = actions.detach().cpu().numpy()
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            # Statistics
            rewards, dones = th.tensor(rewards), th.tensor(dones)
            # reset proprio_buffer if the episode is finished
            self.policy.reset_buffer(dones=dones)

            if th.any(dones == 1):
                n_succ = sum([infos[i]["success"] for i in range(50)])
                self.succ_rate = n_succ / n_envs

            self.step_reward += rewards
            self.step_length += 1
            done_indices = dones.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            self.loss = loss.item()

            not_dones = 1.0 - dones.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.n_steps = n_steps
            self.model.adaptation_steps = n_steps
            self.log_tensorboard()

            if n_steps % 1e4 == 0:
                self.save(os.path.join(self.nn_dir, f"{int(self.n_steps//1e4)}0K"))
                self.save(os.path.join(self.nn_dir, "latest_model"))

            if self.succ_rate >= self.best_succ_rate:
                self.save(os.path.join(self.nn_dir, "best_model"))
                self.best_succ_rate = self.succ_rate
                self.model.best_succ_rate = self.best_succ_rate

            all_fps = self.num_timesteps / (time.time() - _t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(n_steps // 1e3):04}k | FPS: {all_fps:.1f} | "
                f"Current Loss: {loss.item():.5f} | "
                f"Succ. Rate: {self.succ_rate:.5f} | "
                f"Best Succ. Rate: {self.best_succ_rate:.5f}"
            )
            tprint(info_string)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self._last_obs = new_obs
            self.num_timesteps += self.env.num_envs

    def log_tensorboard(self):
        self.logger.add_scalar("adaptation_training/loss", self.loss, self.n_steps)
        self.logger.add_scalar(
            "adaptation_training/mean_episode_rewards",
            self.mean_eps_reward.get_mean(),
            self.n_steps,
        )
        self.logger.add_scalar(
            "adaptation_training/mean_episode_length",
            self.mean_eps_length.get_mean(),
            self.n_steps,
        )
        self.logger.add_scalar(
            "adaptation_training/success_rate", self.succ_rate, self.n_steps
        )

    def save(self, name):
        self.model.save(name)

    def compute_mean_adaptor_loss(self):
        """Compute the mean adapter loss over a short rollout without updating weights."""
        n_steps = 0
        self.succ_rate = 0

        _t = time.time()
        _last_t = time.time()
        self._last_obs = self.env.reset()
        n_envs = self.env.num_envs
        losses = []
        assert self._last_obs is not None, "No previous observation was provided"
        while n_steps <= 200 * 10:
            # Convert to pytorch tensor
            obs_tensor = obs_as_tensor(self._last_obs, self.device)

            for key, tensor in obs_tensor.items():
                obs_tensor[key] = tensor.detach()
            actions, _, _, e, e_gt = self.policy(obs_tensor, adapt_trn=True)

            loss = ((e - e_gt.detach()) ** 2).mean()
            losses.append(loss.item())

            # Rescale and perform action
            actions = actions.detach().cpu().numpy()
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )
            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            # Statistics
            rewards, dones = th.tensor(rewards), th.tensor(dones)
            # reset proprio_buffer if the episode is finished
            self.policy.reset_buffer(dones=dones)

            if th.any(dones == 1):
                n_succ = sum([infos[i]["success"] for i in range(50)])
                self.succ_rate = n_succ / n_envs

            self.step_reward += rewards
            self.step_length += 1
            done_indices = dones.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            self.loss = loss.item()

            not_dones = 1.0 - dones.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.n_steps = n_steps
            self.model.adaptation_steps = n_steps

            if self.succ_rate >= self.best_succ_rate:
                self.save(os.path.join(self.nn_dir, "best_model"))
                self.best_succ_rate = self.succ_rate
                self.model.best_succ_rate = self.best_succ_rate

            all_fps = self.num_timesteps / (time.time() - _t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(n_steps // 1e3):04}k | FPS: {all_fps:.1f} | "
                f"Current Loss: {loss.item():.5f} | "
                f"Succ. Rate: {self.succ_rate:.5f} | "
                f"Best Succ. Rate: {self.best_succ_rate:.5f}"
            )
            tprint(info_string)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self._last_obs = new_obs
            self.num_timesteps += self.env.num_envs
        return np.mean(losses)
