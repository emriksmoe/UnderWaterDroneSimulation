""" 
DQN single agent training module.
"""

import os
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

from src.rl.environments.single_agent_env import DTNDroneEnvironment
from src.config.simulation_config import SimulationConfig

class AoILoggingCallback:
    """Custom callback to log AoI-specific metrics during training"""
    
    def __init__(self, log_freq=100):
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_aoi_metrics = []
        
    def __call__(self, locals_, globals_):
        # This gets called at each step
        if locals_.get('done') or locals_.get('truncated'):
            self.episode_count += 1
            
            # Get episode info
            info = locals_.get('infos', [{}])[0]
            episode_reward = locals_.get('episode_reward', 0)
            
            self.episode_rewards.append(episode_reward)
            
            # Log AoI metrics if available
            if 'aoi_metrics' in info:
                aoi_data = info['aoi_metrics']
                self.episode_aoi_metrics.append(aoi_data)
                
                if self.episode_count % self.log_freq == 0:
                    delivered_count = aoi_data['delivered']['count']
                    delivery_rate = aoi_data['global']['delivery_rate']
                    mean_delivery_aoi = aoi_data['delivered']['mean_aoi']
                    
                    print(f"\n📊 Episode {self.episode_count} AoI Metrics:")
                    print(f"  💌 Messages delivered: {delivered_count}")
                    print(f"  📈 Delivery rate: {delivery_rate:.1%}")
                    print(f"  ⏱️  Mean delivery AoI: {mean_delivery_aoi:.1f}s")
                    print(f"  🎯 Episode reward: {episode_reward:.2f}")
        
        return True

def create_environment(config: SimulationConfig):
    """Create and wrap the environment for training"""
    env = DTNDroneEnvironment(config)
    
    # Check environment is compatible with SB3
    check_env(env)
    print("✅ Environment check passed!")
    
    # Wrap with Monitor for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/training_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = Monitor(env, log_dir)
    
    return env, log_dir

def create_dqn_model(env, config: SimulationConfig):
    """Create DQN model with AoI-optimized hyperparameters"""
    
    model = DQN(
        policy="MlpPolicy",
        env=env,
        
        # Network architecture
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=torch.nn.ReLU,
        ),
        
        # Learning parameters
        learning_rate=1e-4,
        gamma=0.95,
        
        # Experience replay
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        
        # Target network updates
        target_update_interval=1000,
        tau=1.0,
        
        # Exploration strategy
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        
        # Training frequency
        train_freq=4,
        gradient_steps=1,
        
        # Other parameters
        verbose=1,
        tensorboard_log="./logs",  # ← ADD THIS LINE
        device="auto",
        seed=42,
    )
    
    print(f"🧠 Created DQN model:")
    print(f"  📊 Observation space: {env.observation_space.shape}")
    print(f"  🎮 Action space: {env.action_space.n} actions")
    print(f"  🔧 Network architecture: [256, 256, 128]")
    print(f"  📚 Buffer size: {model.buffer_size}")
    print(f"  📈 TensorBoard log: ./logs")  # ← ADD THIS LINE
    
    return model

def create_callbacks(env, log_dir: str):
    """Create training callbacks for evaluation and logging"""
    
    # Evaluation callback - saves best model
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=2000,              # Evaluate every 2000 steps
        deterministic=True,           # Use deterministic policy for evaluation
        render=False,
        n_eval_episodes=5,           # Number of episodes for evaluation
        verbose=1
    )
    
    # Custom AoI logging callback
    aoi_callback = AoILoggingCallback(log_freq=50)
    
    return CallbackList([eval_callback])

def train_dqn_agent():
    """Main training function"""
    
    print("🚀 Starting DQN Training for DTN Drone AoI Optimization")
    print("=" * 60)
    
    # Create configuration
    config = SimulationConfig()
    
    # Print key configuration
    print(f"📋 Training Configuration:")
    print(f"  🌊 Area size: {config.area_size}")
    print(f"  📡 Sensors: {config.num_sensors}")
    print(f"  🚢 Ships: {config.num_ships}")
    print(f"  ⏱️  Data generation interval: {config.data_generation_interval}s")
    print(f"  📦 Message TTL: {config.message_ttl}s")
    print(f"  🔄 Max episode steps: {config.max_episode_steps}")
    print()
    
    # Create environment
    env, log_dir = create_environment(config)
    
    # Create DQN model
    model = create_dqn_model(env, config)
    
    # Create callbacks
    callbacks = create_callbacks(env, log_dir)
    
    # Start training
    print("\n🎯 Beginning Training...")
    print("Monitor training progress with: tensorboard --logdir ./logs")
    
    try:
        model.learn(
            total_timesteps=100000,      # Total training steps
            callback=callbacks,
            log_interval=100,            # Print progress every 100 episodes
            tb_log_name="dqn_aoi_training",
            progress_bar=True
        )
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_trained_model(model, env, episodes=3)
    
    env.close()
    return model, log_dir

def test_trained_model(model, env, episodes=3):
    """Test the trained model and show performance"""
    
    print(f"\n🔬 Testing trained model for {episodes} episodes...")
    
    total_rewards = []
    total_delivered = []
    total_delivery_rates = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n📝 Episode {episode + 1}:")
        
        while True:
            # Use trained policy (deterministic)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        
        # Extract AoI metrics
        if 'aoi_metrics' in info:
            aoi_data = info['aoi_metrics']
            delivered_count = aoi_data['delivered']['count']
            delivery_rate = aoi_data['global']['delivery_rate']
            mean_aoi = aoi_data['delivered']['mean_aoi']
            
            total_delivered.append(delivered_count)
            total_delivery_rates.append(delivery_rate)
            
            print(f"  🎯 Steps: {step_count}, Reward: {episode_reward:.2f}")
            print(f"  📦 Messages delivered: {delivered_count}")
            print(f"  📈 Delivery rate: {delivery_rate:.1%}")
            print(f"  ⏱️  Mean delivery AoI: {mean_aoi:.1f}s")
    
    # Summary statistics
    print(f"\n📊 Test Summary ({episodes} episodes):")
    print(f"  🎯 Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  📦 Mean messages delivered: {np.mean(total_delivered):.1f}")
    print(f"  📈 Mean delivery rate: {np.mean(total_delivery_rates):.1%}")

def compare_with_random_policy(config: SimulationConfig, episodes=5):
    """Compare trained model with random policy"""
    
    print("\n🎲 Testing random policy for comparison...")
    
    env = DTNDroneEnvironment(config)
    random_rewards = []
    random_delivered = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        random_rewards.append(episode_reward)
        
        if 'aoi_metrics' in info:
            delivered_count = info['aoi_metrics']['delivered']['count']
            random_delivered.append(delivered_count)
    
    print(f"🎲 Random Policy Results ({episodes} episodes):")
    print(f"  🎯 Mean reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"  📦 Mean messages delivered: {np.mean(random_delivered):.1f}")
    
    env.close()

if __name__ == "__main__":
    # Train the DQN agent
    model, log_dir = train_dqn_agent()
    
    # Compare with random policy
    config = SimulationConfig()
    compare_with_random_policy(config, episodes=3)
    
    print(f"\n🎉 Training complete! Check logs at: {log_dir}")
    print("📈 View training progress: tensorboard --logdir ./logs")