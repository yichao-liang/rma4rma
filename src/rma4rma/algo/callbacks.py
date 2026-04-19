import os

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from rma4rma.algo.evaluate_policy import evaluate_policy


class CheckpointCallbackRMA(CheckpointCallback):

    def _latest_ckpt_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_" or "vecnormalize_"
            for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(
            self.save_path, f"{self.name_prefix}_{checkpoint_type}latest.{extension}"
        )

    def _on_step(self) -> bool:
        if self.n_calls % (self.save_freq // 10) == 0:
            model_path = self._latest_ckpt_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 1:
                print(f"Saving latest model checkpoint to {model_path}")

            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                replay_buffer_path = self._latest_ckpt_path(
                    "replay_buffer_", extension="pkl"
                )
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(
                        f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                    )

            if (
                self.save_vecnormalize
                and self.model.get_vec_normalize_env() is not None
            ):
                # Save the VecNormalize statistics
                vec_normalize_path = self._latest_ckpt_path(
                    "vecnormalize_", extension="pkl"
                )
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return super()._on_step()


class EvalCallbackRMA(EvalCallback):

    def __init__(self, *args, **kwargs):
        self.best_success_rate = -np.inf
        self.num_envs = kwargs.pop("num_envs")
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer: list = []
            success_rate = 0.0

            # update eval_env's step counter
            self.eval_env.env_method(
                "set_step_counter", self.model.num_timesteps // self.num_envs
            )
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if success_rate >= self.best_success_rate:
                if self.verbose >= 1:
                    print("New best success rate!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_success_rate = success_rate
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
