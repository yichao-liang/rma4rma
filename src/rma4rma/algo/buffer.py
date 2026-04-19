from typing import Generator, Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.type_aliases import DictRolloutBufferSamples


class DictRolloutBufferRMA(DictRolloutBuffer):
    """A buffer that also produce proprioception hisotry from being sampled."""

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        """Modify the `get` method to prepare a buffer of the proprioception history
        which will have shape (num_envs*rollout_size, prop_buffer_size, proprio_dim)"""
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Begin modification
        if self.use_prop_act_history:
            # rollout_size is stored in self.buffer_size
            self.prop_history_size = 50
            self.prop_dim = (
                self.observations["agent_state"].shape[1] + self.actions.shape[1]
            )
            self.n_history = self.buffer_size * self.n_envs
            self.proprio_history = th.zeros(
                self.n_history,
                self.prop_history_size,
                self.prop_dim,
                device=self.device,
            )
            # self.observations['agent_state'] and self.actions contains data
            # collected from multiple epsiodes, each with 50 timesteps.
            # For the proprio_history, each entry correspondes to time t in a 50
            #   step episode. We want each history to be filled with 50 - t rows of
            #   zeros at the beginning and the proprioception history for t steps
            # Each datapoint's t in each episode (number of data in each hisotry)
            # [0, 1, 2, ..., 49, 0, 1, 2, ..., 49]
            ind_t = th.range(0, self.n_history - 1).int() % 50

            observations = self.to_torch(self.observations["agent_state"])
            actions = self.to_torch(self.actions)

            # Create a 2D tensor where each row starts from the corresponding
            # ind_buff_start value and goes up to 49
            mask = th.arange(50).unsqueeze(0).repeat(
                self.n_history, 1
            ) < ind_t.unsqueeze(1)

            # Repeat the proprioception history for each episode 50 times and
            #   select the rows for each datapoint
            observations = observations.reshape(self.n_history // 50, 50, -1)
            observations = observations.repeat_interleave(50, dim=0)
            actions = actions.reshape(self.n_history // 50, 50, -1)
            actions = actions.repeat_interleave(50, dim=0)
            self.proprio_history[mask.flip([1])] = th.cat(
                [observations, actions], dim=2
            )[mask]

            # add to the observations
            self.observations["prop_act_history"] = self.proprio_history
        # End modification

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
