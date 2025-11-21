""" 
DQN single agent training module.
"""

import os
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

from src.rl.environments.single_agent_env import DTNDroneEnvironment
from src.config.simulation_config import SimulationConfig
from src.utils.config_logger import save_training_config


class AoILoggingCallback(BaseCallback):
    """Custom callback to log AoI-specific metrics during training"""
    
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_aoi_metrics = []

    def _init_callback(self) -> None:
        """Called when callback is initialized"""
        pass
        
    def _on_step(self) -> bool:
        """
        Called at every step.
        Returns True to continue training, False to stop.
        """
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Get info from the environment
            infos = self.locals.get('infos', [{}])
            if infos and len(infos) > 0:
                info = infos[0]
                
                # Extract episode reward if available
                episode_reward = 0
                if 'episode' in self.locals:
                    episode_reward = self.locals['episode']['r']
                    self.episode_rewards.append(episode_reward)
                
                # Extract AoI metrics if available
                if 'aoi_metrics' in info:
                    aoi_data = info['aoi_metrics']
                    self.episode_aoi_metrics.append(aoi_data)
                    
                    # Log every log_freq episodes
                    if self.episode_count % self.log_freq == 0:
                        self._log_episode_metrics(info, episode_reward)
        
        return True  # Continue training
    
    def _log_episode_metrics(self, info, episode_reward):
        """Log detailed AoI metrics"""
        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} AoI Metrics:")
        print(f"{'='*60}")
        
        if 'aoi_metrics' in info:
            metrics = info['aoi_metrics']
            
            # Delivered messages
            if 'delivered' in metrics:
                delivered = metrics['delivered']
                print(f"  Delivered Messages: {delivered.get('count', 0)}")
                print(f"  Mean Delivery AoI: {delivered.get('mean_aoi', 0):.1f}s")
                print(f"  Min/Max AoI: {delivered.get('min_aoi', 0):.1f}s / {delivered.get('max_aoi', 0):.1f}s")
            
            # Undelivered messages
            if 'undelivered' in metrics:
                undelivered = metrics['undelivered']
                print(f"  Undelivered Messages: {undelivered.get('count', 0)}")
                print(f"    In Sensors: {undelivered.get('in_sensors', 0)}")
                print(f"    In Drone: {undelivered.get('in_drone', 0)}")
                print(f"  Mean Undelivered AoI: {undelivered.get('mean_current_aoi', 0):.1f}s")
            
            # Global metrics
            if 'global' in metrics:
                global_m = metrics['global']
                print(f"  Delivery Rate: {global_m.get('delivery_rate', 0)*100:.1f}%")
                print(f"  Total System AoI: {global_m.get('system_aoi', 0):.1f}s")
        
        # Episode statistics
        if 'stats' in info:
            stats = info['stats']
            print(f"\nEpisode Statistics:")
            print(f"  Messages Collected: {stats.get('messages_collected', 0)}")
            print(f"  Messages Delivered: {stats.get('messages_delivered', 0)}")
            print(f"  Distance Traveled: {stats.get('total_distance_traveled', 0):.1f}m")
            print(f"  Sensor Visits: {stats.get('sensor_visits', 0)}")
            print(f"  Ship Visits: {stats.get('ship_visits', 0)}")
        
        # Recent rewards
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-self.log_freq:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"\nAverage Reward (last {len(recent_rewards)} episodes): {avg_reward:.1f}")
        
        print(f"{'='*60}\n")


def get_model_name():
    """
    Prompt user for model name or generate timestamped name.
    Returns clean folder-safe name.
    """
    print("\n" + "="*60)
    print("📝 MODEL NAMING")
    print("="*60)
    print("\nEnter a name for this training run (or press Enter for auto-name):")
    print("Examples: 'baseline_v1', 'high_penalties', 'optimized_final'")
    print("(Use letters, numbers, underscores, and hyphens only)")
    
    user_input = input("\nModel name: ").strip()
    
    if user_input:
        # Clean the name (remove invalid characters)
        clean_name = "".join(c for c in user_input if c.isalnum() or c in "_-")
        
        # Add timestamp suffix for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{clean_name}_{timestamp}"
        
        print(f"\n✅ Model will be saved as: {model_name}")
    else:
        # Auto-generate name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"training_{timestamp}"
        print(f"\n✅ Auto-generated name: {model_name}")
    
    return model_name


def save_model_metadata(log_dir: str, model_name: str, config: SimulationConfig):
    """Save metadata about the trained model"""
    import json
    
    metadata = {
        'model_name': model_name,
        'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_timesteps': config.dqn_total_timesteps,
        'log_directory': log_dir
    }
    
    metadata_path = Path(log_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Model metadata saved to: {metadata_path}")


def create_environment(config: SimulationConfig, log_dir: str):
    """Create and wrap the environment for training"""
    env = DTNDroneEnvironment(config)
    
    # Check environment is compatible with SB3
    check_env(env)
    print("✅ Environment check passed!")
    
    # Wrap with Monitor for logging
    env = Monitor(env, log_dir)
    
    return env


def create_dqn_model(env, config: SimulationConfig):
    """Create DQN model with AoI-optimized hyperparameters"""
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        
        # Network architecture
        policy_kwargs=dict(
            net_arch=config.dqn_net_arch,
            activation_fn=getattr(torch.nn, config.dqn_activation),
        ),
        
        # Learning parameters
        learning_rate=config.dqn_learning_rate,
        gamma=config.dqn_gamma,
        
        # Experience replay
        buffer_size=config.dqn_buffer_size,
        learning_starts=config.dqn_learning_starts,
        batch_size=config.dqn_batch_size,
        
        # Target network updates
        target_update_interval=config.dqn_target_update_interval,
        tau=config.dqn_tau,
        
        # Exploration strategy
        exploration_fraction=config.dqn_exploration_fraction,
        exploration_initial_eps=config.dqn_exploration_initial_eps,
        exploration_final_eps=config.dqn_exploration_final_eps,
        
        # Training frequency
        train_freq=config.dqn_train_freq,
        gradient_steps=config.dqn_gradient_steps,
        
        # Other parameters
        verbose=config.dqn_verbose,
        tensorboard_log="./logs/tensorboard_logs",
        device="auto",
        seed=config.dqn_seed,
    )
    
    print(f"\n✅ Created DQN model:")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n} actions")
    print(f"   Network architecture: {config.dqn_net_arch}")
    print(f"   Buffer size: {model.buffer_size:,}")
    print(f"   Learning rate: {config.dqn_learning_rate}")
    
    return model


def create_callbacks(env, log_dir: str):
    """Create training callbacks for evaluation and logging"""
    
    # Evaluation callback - saves best model
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=2000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    # Custom AoI logging callback
    aoi_callback = AoILoggingCallback(log_freq=200, verbose=0)
    
    return CallbackList([eval_callback, aoi_callback])


def train_dqn_agent():
    """Main training function"""
    
    print("\n" + "="*60)
    print("🚁 DQN Training - DTN Drone AoI Optimization")
    print("="*60)

    # Get model name from user
    model_name = get_model_name()
    
    # Create named log directory
    log_dir = f"./logs/single_agent/{model_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n📂 Log directory: {log_dir}")
    
    # Create configuration
    config = SimulationConfig()
    
    # Save configuration BEFORE training
    save_training_config(config, log_dir, model_name)
    
    # Print key configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   Area size: {config.area_size}")
    print(f"   Sensors: {config.num_sensors}")
    print(f"   Ships: {config.num_ships}")
    print(f"   Network: {config.dqn_net_arch}")
    print(f"   Buffer: {config.dqn_buffer_size:,}")
    print(f"   Learning rate: {config.dqn_learning_rate}")
    print(f"   Total timesteps: {config.dqn_total_timesteps:,}")
    print("="*60)
    
    # Create environment
    env = create_environment(config, log_dir)
    
    # Create DQN model
    model = create_dqn_model(env, config)
    
    # Create callbacks
    callbacks = create_callbacks(env, log_dir)
    
    # Start training
    print("\n🚀 Beginning Training...")
    print("Monitor progress: tensorboard --logdir ./logs")
    
    try:
        model.learn(
            total_timesteps=config.dqn_total_timesteps,
            callback=callbacks,
            log_interval=config.dqn_log_interval,
            tb_log_name=config.dqn_tb_log_name,
            progress_bar=True
        )
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    final_model_path = f"{log_dir}/final_model.zip"
    model.save(final_model_path)
    print(f"\n💾 Final model saved: {final_model_path}")
    
    # Save metadata
    save_model_metadata(log_dir, model_name, config)
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_trained_model(model, env, config.test_episodes)
    
    env.close()
    
    print(f"\n{'='*60}")
    print("🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"📁 Model directory: {log_dir}")
    print(f"📊 TensorBoard: tensorboard --logdir ./logs")
    print(f"💾 Best model: {log_dir}/best_model/best_model.zip")
    print(f"📝 Config saved: {log_dir}/config.json")
    print(f"\n🔬 Test your model:")
    print(f"   python -m src.rl.testing.test_single_agent_model")
    print(f"{'='*60}\n")
    
    return model, log_dir


def test_trained_model(model, env, episodes):
    """Test the trained model and show performance"""
    
    print(f"\nTesting for {episodes} episodes...")
    
    total_rewards = []
    total_delivered = []
    total_delivery_rates = []
    total_mean_aois = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        
        if 'aoi_metrics' in info:
            aoi_data = info['aoi_metrics']
            delivered_count = aoi_data['delivered']['count']
            delivery_rate = aoi_data['global']['delivery_rate']
            mean_aoi = aoi_data['delivered']['mean_aoi']
            
            total_delivered.append(delivered_count)
            total_delivery_rates.append(delivery_rate)
            total_mean_aois.append(mean_aoi)
    
    print(f"\n📊 Test Summary ({episodes} episodes):")
    print(f"   Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"   Messages Delivered: {np.mean(total_delivered):.1f} ± {np.std(total_delivered):.1f}")
    print(f"   Delivery Rate: {np.mean(total_delivery_rates):.1%}")
    print(f"   Mean AoI: {np.mean(total_mean_aois):.1f}s ± {np.std(total_mean_aois):.1f}s")


if __name__ == "__main__":
    train_dqn_agent()