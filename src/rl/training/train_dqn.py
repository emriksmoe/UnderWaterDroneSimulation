# train_dqn.py

import os
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.environments.rl_env import DroneAoIEnv
from src.config.simulation_config import SimulationConfig


# -------------------------
# Environment factory (NO ActionMasker for DQN)
# -------------------------
def make_env():
    config = SimulationConfig()
    env = DroneAoIEnv(
        config=config, 
        episode_duration=86400     # No ship visit penalty
    )
    return env


# -------------------------
# Custom evaluation callback
# -------------------------
class AoIEvalCallback(BaseCallback):
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
        os.makedirs(self.best_model_save_path, exist_ok=True)

        if getattr(self.eval_env, "num_envs", 1) != 1:
            raise ValueError(
                f"AoIEvalCallback expects eval_env.num_envs == 1, got {self.eval_env.num_envs}."
            )

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, mean_len, infos = self._run_evaluation()

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", float(mean_len))

            if len(infos) > 0:
                self._log_info_metrics(infos)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = float(mean_reward)
                save_path = os.path.join(self.best_model_save_path, "best_model")
                self.model.save(save_path)
                if self.verbose:
                    print(f"[Eval] New best mean_reward={mean_reward:.4f} -> saved to {save_path}.zip")

        return True

    def _run_evaluation(self):
        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        ep_infos: List[Dict[str, Any]] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            total_r = 0.0
            steps = 0

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)

                total_r += float(reward[0])
                steps += 1

                if done[0]:
                    info0 = info[0] if isinstance(info, (list, tuple)) else info
                    if isinstance(info0, dict) and info0:
                        ep_infos.append(info0)

            ep_rewards.append(total_r)
            ep_lengths.append(steps)

        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        return mean_reward, mean_len, ep_infos

    def _log_info_metrics(self, infos: List[Dict[str, Any]]):
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
            self.logger.record("eval/generated", float(np.mean(generated)))
        if delivery_rate:
            self.logger.record("eval/delivery_rate_direct", float(np.mean(delivery_rate)))


# -------------------------
# Main training
# -------------------------
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # DQN uses single env
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Folders
    tb_dir = f"./tensorboard_logs/dqn_{timestamp}/"
    best_dir = f"./models/dqn_{timestamp}/"
    ckpt_dir = f"./checkpoints/dqn_{timestamp}/"
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        policy_kwargs=dict(net_arch=[256, 256]),
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=128,
        gamma=0.995,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=tb_dir,
    )

    eval_callback = AoIEvalCallback(
        eval_env=eval_env,
        eval_freq=50_000,
        n_eval_episodes=3,
        best_model_save_path=best_dir,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix="dqn_drone",
    )

    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(best_dir, "dqn_drone_final"))
    print(f"Final model saved to {os.path.join(best_dir, 'dqn_drone_final.zip')}")