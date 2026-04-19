import random
from collections import OrderedDict

import gymnasium.spaces as spaces
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from rma4rma.algo.buffer import DictRolloutBufferRMA


class PPORMA(PPO):

    def __init__(
        self, *args, auto_dr=False, eval=False, use_prop_history_base=False, **kwargs
    ):

        self.use_prop_history_base = use_prop_history_base
        super().__init__(*args, **kwargs)
        if use_prop_history_base:
            self.n_history = self.n_envs
            self.prop_history_size = 50
            self.prop_dim = (
                self.action_space.shape[0]
                + self.observation_space["agent_state"].shape[0]
            )
            self.proprio_history = np.zeros(
                [self.n_history, self.prop_history_size, self.prop_dim]
            )
        # Automatic Domain Randomization (ADR) init
        self.auto_dr = auto_dr
        if auto_dr:
            # to record the eval results
            self.succ_queue = []
            self.succ_rate_l = 0.01
            self.succ_rate_h = 0.1
            # determines which envs are used for evaluation
            self.eval_env_id = list(range(20))
            if not eval:
                self.env.set_attr("eval_env", True, self.eval_env_id)
            self.dr_params_init = OrderedDict(
                obj_scale=[1.0, 1.0],
                obj_density=[1.0, 1.0],
                obj_friction=[1.0, 1.0],
                force_scale=[0.0, 0.0],
                obj_position=[0.0, 0.0],
                obj_rotation=[0.0, 0.0],
                prop_position=[0.0, 0.0],
            )
            self.dr_params_now = OrderedDict(
                obj_scale=[1.0, 1.0],
                obj_density=[1.0, 1.0],
                obj_friction=[1.0, 1.0],
                force_scale=[0.0, 0.0],
                obj_position=[0.0, 0.0],
                obj_rotation=[0.0, 0.0],
                prop_position=[0.0, 0.0],
            )
            self.dr_params_end = OrderedDict(
                obj_scale=[0.7, 1.2],
                obj_density=[0.5, 5.0],
                obj_friction=[0.5, 1.1],
                force_scale=[0.0, 2.0],
                obj_position=[-0.005, 0.005],
                obj_rotation=[-np.pi * (1 / 18), np.pi * (1 / 18)],
                prop_position=[-0.005, 0.005],
            )
            self.dr_delta = OrderedDict(
                obj_scale=0.1,
                obj_density=0.1,
                obj_friction=0.1,
                force_scale=0.1,
                obj_position=0.002,
                obj_rotation=0.002,
                prop_position=0.002,
            )
            self.randomized_param = random.choice(list(self.dr_params_init.keys()))
            if not eval:
                self.env.set_attr(
                    "randomized_param", self.randomized_param, self.eval_env_id
                )

    def _setup_model(self) -> None:
        """Modified to use `DictRolloutBufferRMA` instead of `DictRolloutBuffer`"""
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBufferRMA

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.rollout_buffer.use_prop_act_history = self.use_prop_history_base

        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

    def test_eval(self, expert_adapt=False, only_dr=False, without_adapt_module=False):
        self.policy.test_eval(
            expert_adapt=expert_adapt,
            only_dr=only_dr,
            without_adapt_module=without_adapt_module,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Modified based on the original `collect_rollouts`:
        1. pass the `dones` variable to the `predict_values` to select the
          corresponding action;
        2. reset the `prev_action` in the `policy` after each episode is done.

        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                if self.use_prop_history_base:
                    self._last_obs["prop_act_history"] = self.proprio_history
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # update the proprioception history with the new state and action
            if self.use_prop_history_base:
                state_action_vec = np.concatenate(
                    [self._last_obs["agent_state"], actions], axis=1
                )[:, np.newaxis]
                self.proprio_history = np.concatenate(
                    [self.proprio_history[:, 1:], state_action_vec], axis=1
                )

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with the value function.
            # See https://github.com/DLR-RM/stable-baselines3/issues/633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    if self.use_prop_history_base:
                        terminal_obs["prop_act_history"] = th.tensor(
                            self.proprio_history[idx : idx + 1], device=self.device
                        )
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs, done_idx=idx
                        )[0]
                    self.policy.reset_prev_action(done_idx=idx)
                    if self.use_prop_history_base:
                        self.proprio_history = np.zeros(
                            [self.n_history, self.prop_history_size, self.prop_dim]
                        )
                    rewards[idx] += self.gamma * terminal_value
                    if self.auto_dr:
                        if idx in self.eval_env_id:
                            self.succ_queue.append(infos[idx]["success"])
                        self.adr_update()

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            if self.use_prop_history_base:
                new_obs["prop_act_history"] = self.proprio_history
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
            if self.use_prop_history_base:
                self.proprio_history = np.zeros(
                    [self.n_history, self.prop_history_size, self.prop_dim]
                )
            self.policy.reset_prev_action(done_idx=idx)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def adr_update(self):
        if len(self.succ_queue) >= 500:
            succ_rate = np.mean(self.succ_queue)
            if succ_rate > self.succ_rate_h:
                # widen the range of the randomized param
                new_range_l = max(
                    self.dr_params_now[self.randomized_param][0]
                    - self.dr_delta[self.randomized_param],
                    self.dr_params_end[self.randomized_param][0],
                )
                new_range_h = min(
                    self.dr_params_now[self.randomized_param][1]
                    + self.dr_delta[self.randomized_param],
                    self.dr_params_end[self.randomized_param][1],
                )
                print(
                    f"randomized_param: {self.randomized_param}, succ_rate: {succ_rate}, widen the range"
                )
                print(
                    f"old range: {self.dr_params_now[self.randomized_param]}, new range: [{new_range_l}, {new_range_h}]"
                )
                self.dr_params_now[self.randomized_param] = [new_range_l, new_range_h]
            if succ_rate < self.succ_rate_l:
                # narrow the range of the randomized param
                new_range_l = min(
                    self.dr_params_now[self.randomized_param][0]
                    + self.dr_delta[self.randomized_param],
                    self.dr_params_init[self.randomized_param][0],
                )
                new_range_h = max(
                    self.dr_params_now[self.randomized_param][1]
                    - self.dr_delta[self.randomized_param],
                    self.dr_params_init[self.randomized_param][1],
                )
                print(
                    f"randomized_param: {self.randomized_param}, succ_rate: {succ_rate}, narrow the range"
                )
                print(
                    f"old range: {self.dr_params_now[self.randomized_param]}, new range: [{new_range_l}, {new_range_h}]"
                )
                self.dr_params_now[self.randomized_param] = [new_range_l, new_range_h]
            # Push the updated DR ranges to the envs and pick the next parameter.
            self.env.set_attr("dr_params", self.dr_params_now)
            self.succ_queue = []
            self.randomized_param = random.choice(list(self.dr_params_init.keys()))
            self.env.set_attr(
                "randomized_param", self.randomized_param, self.eval_env_id
            )
