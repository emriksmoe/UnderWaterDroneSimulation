# train_rl.py

import os
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.environments.rl_env import DroneAoIEnv
from src.config.simulation_config import SimulationConfig


# -------------------------
# Environment factory
# -------------------------
def make_env():
    config = SimulationConfig()
    #set shaping lambda and dither here, set to 0 for pure AoI reward
    #lambda should be 10 and dither 1e-3 for reward shaping
    env = DroneAoIEnv(config=config, episode_duration=86400, warmup_duration = 3600)
    # 24 hours SimPy time per episode
    return ActionMasker(env, lambda env: env.action_masks())


# -------------------------
# Custom evaluation callback
# (replaces EvalCallback so we do NOT evaluate twice)
# -------------------------
class AoIEvalCallback(BaseCallback):
    """
    Runs evaluation every eval_freq steps:
      - saves best model (by mean reward)
      - logs custom metrics from info dict at episode end
    """

    def __init__(
        self,
        eval_env: DummyVecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic

        self.best_mean_reward = -np.inf

        # Ensure folder exists
        os.makedirs(self.best_model_save_path, exist_ok=True)

        # IMPORTANT: This evaluator is written for a single-env VecEnv
        if getattr(self.eval_env, "num_envs", 1) != 1:
            raise ValueError(
                f"AoIEvalCallback expects eval_env.num_envs == 1, got {self.eval_env.num_envs}."
                "Use DummyVecEnv([make_env]) for eval."
            )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        # Trigger evaluation based on num_timesteps (not n_calls)
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, mean_len, infos = self._run_evaluation()

            # --- SB3-style reward logging ---
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", float(mean_len))

            # --- Your baseline-aligned metrics (from info dict) ---
            if len(infos) > 0:
                self._log_info_metrics(infos)

            # Save best model (by mean reward)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = float(mean_reward)
                save_path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(save_path)
                if self.verbose:
                    print(
                        f"[Eval] New best mean_reward={mean_reward:.4f} -> saved to {save_path}.zip"
                    )

        return True

    def _run_evaluation(self):
        """
        Runs n_eval_episodes episodes on eval_env (single env),
        collecting episode rewards/lengths and final info dict.
        """
        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        ep_infos: List[Dict[str, Any]] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])

            total_r = 0.0
            steps = 0

            while not done[0]:
                mask = self.eval_env.envs[0].action_masks()
                action, _ = self.model.predict(
                    obs,
                    action_masks=np.array([mask]),
                    deterministic=self.deterministic,
                )
                obs, reward, done, info = self.eval_env.step(action)

                # DummyVecEnv returns arrays
                total_r += float(reward[0])
                steps += 1

                if done[0]:
                    # info is a list of dicts (one per env)
                    info0 = info[0] if isinstance(info, (list, tuple)) else info
                    if isinstance(info0, dict) and info0:
                        ep_infos.append(info0)

            ep_rewards.append(total_r)
            ep_lengths.append(steps)

        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        return mean_reward, mean_len, ep_infos

    def _log_info_metrics(self, infos: List[Dict[str, Any]]):
        """
        Logs mean across evaluation episodes for metrics you care about.
        """
        def grab(key: str) -> List[float]:
            vals = []
            for d in infos:
                if key in d:
                    try:
                        vals.append(float(d[key]))
                    except Exception:
                        pass
            return vals

        aoi_int = grab("aoi_integral")
        time_avg = grab("time_avg_aoi")
        delivered = grab("delivered")
        expired = grab("expired")
        generated = grab("generated")
        delivery_rate = grab("delivery_rate")
        dropped_sensor = grab("dropped_sensor_buffer")
        dropped_drone = grab("dropped_drone_buffer")

        if dropped_sensor:
            self.logger.record("eval/dropped_sensor_buffer", float(np.mean(dropped_sensor)))
        if dropped_drone:
            self.logger.record("eval/dropped_drone_buffer", float(np.mean(dropped_drone)))

        if aoi_int:
            self.logger.record("eval/aoi_integral", float(np.mean(aoi_int)))
        if time_avg:
            self.logger.record("eval/time_avg_aoi", float(np.mean(time_avg)))
        if delivered:
            self.logger.record("eval/delivered", float(np.mean(delivered)))
        if expired:
            self.logger.record("eval/expired", float(np.mean(expired)))
        if generated:
            gen_mean = float(np.mean(generated))
            self.logger.record("eval/generated", gen_mean)
        if delivery_rate:
            self.logger.record("eval/delivery_rate_direct", float(np.mean(delivery_rate)))  # Already computed


# -------------------------
# Main training
# -------------------------
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- parallel training envs ---
    n_envs = 4
    train_env = DummyVecEnv([make_env for _ in range(n_envs)])

    # --- single eval env (important) ---
    eval_env = DummyVecEnv([make_env])

    # Folders
    tb_dir = f"./tensorboard_logs/run_{timestamp}/"
    best_dir = f"./models/run_{timestamp}/"
    ckpt_dir = f"./checkpoints/run_{timestamp}/"
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = MaskablePPO(
        policy="MlpPolicy",
        policy_kwargs=dict(net_arch=[256, 256]),
        env=train_env,
        n_steps=4096,         # rollout steps per env
        batch_size=512,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, #standard is 0.01, but possible to change
        verbose=1,
        tensorboard_log=tb_dir,
    )

    # Eval: logs baseline-aligned metrics + saves best model (no double eval)
    eval_callback = AoIEvalCallback(
        eval_env=eval_env,
        eval_freq=50_000,
        n_eval_episodes=3,
        best_model_save_path=best_dir,
        deterministic=True,
        verbose=1,
    )

    # Periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="ppo_drone",
    )

    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(best_dir, "ppo_drone_final"))
    print(f"Final model saved to {os.path.join(best_dir, 'ppo_drone_final.zip')}")
