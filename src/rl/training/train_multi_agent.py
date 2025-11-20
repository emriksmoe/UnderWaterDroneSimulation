"""
Multi-Agent DQN training module for multiple DTN drones.
"""

import os
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch

from src.rl.environments.multi_agent_env import MultiAgentDTNEnvironment
from src.config.simulation_config import SimulationConfig

class MultiAgentAoICallback:
    """Custom callback for multi-agent AoI metrics logging"""
    
    def __init__(self, log_freq=100):
        self.log_freq = log_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.multi_agent_metrics = []
        
    def __call__(self, locals_, globals_):
        # This gets called at each step
        if locals_.get('done') or locals_.get('truncated'):
            self.episode_count += 1
            
            # Get multi-agent episode info
            infos = locals_.get('infos', [{}])
            episode_reward = locals_.get('episode_reward', 0)
            
            self.episode_rewards.append(episode_reward)
            
            # Handle vectorized environments - get first environment's info
            info = infos[0] if isinstance(infos, list) else infos
            
            # Log multi-agent AoI metrics if available
            if 'multi_agent_aoi_metrics' in info:
                ma_metrics = info['multi_agent_aoi_metrics']
                self.multi_agent_metrics.append(ma_metrics)
                
                if self.episode_count % self.log_freq == 0:
                    system_metrics = ma_metrics['system']
                    total_delivered = system_metrics['total_delivered']
                    system_delivery_rate = system_metrics['delivery_rate']
                    mean_system_aoi = system_metrics['mean_delivery_aoi']
                    
                    print(f"\n🚁 Episode {self.episode_count} Multi-Agent Metrics:")
                    print(f"  System messages delivered: {total_delivered}")
                    print(f"  System delivery rate: {system_delivery_rate:.1%}")
                    print(f"  System mean AoI: {mean_system_aoi:.1f}s")
                    print(f"  Total episode reward: {episode_reward:.2f}")
                    
                    # Per-drone breakdown
                    for i, drone_metrics in enumerate(ma_metrics['per_drone']):
                        print(f"  Drone {i}: {drone_metrics['delivered']} delivered, "
                              f"{drone_metrics['collected']} collected")
        
        return True

def make_multi_agent_env(config: SimulationConfig, rank: int, seed: int = 0):
    """Create a single multi-agent environment instance"""
    def _init():
        env = MultiAgentDTNEnvironment(config)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def create_multi_agent_environment(config: SimulationConfig, n_envs: int = 1):
    """Create multi-agent environment for training"""
    
    # Test single environment first
    test_env = MultiAgentDTNEnvironment(config)
    check_env(test_env)
    print("Multi-agent environment check passed!")
    
    # Create organized logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/multi_agent/training_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # For multi-agent, we typically use single environment due to complexity
    if n_envs == 1:
        env = MultiAgentDTNEnvironment(config)
        env = Monitor(env, log_dir)
    else:
        # Vectorized environments (optional for advanced users)
        env = SubprocVecEnv([
            make_multi_agent_env(config, i, config.dqn_seed) 
            for i in range(n_envs)
        ])
    
    return env, log_dir, test_env


def create_multi_agent_ppo_model(env, config: SimulationConfig):
    """Create PPO model for multi-agent training (supports MultiDiscrete)"""
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        
        # Network architecture for multi-agent
        policy_kwargs=dict(
            net_arch=[512, 512, 256, 128],
            activation_fn=getattr(torch.nn, config.dqn_activation),
        ),
        
        # PPO specific parameters
        learning_rate=config.dqn_learning_rate,
        gamma=config.dqn_gamma,
        
        # PPO uses different parameters than DQN
        n_steps=2048,           # Steps per environment per update
        batch_size=64,          # Batch size for training
        n_epochs=10,            # Number of epochs per update
        
        # Other parameters
        verbose=config.dqn_verbose,
        tensorboard_log="./logs/tensorboard_logs",
        device="auto",
        seed=config.dqn_seed,
    )
    
    print(f"🧠 Created Multi-Agent PPO model:")
    print(f"  📊 Observation space: {env.observation_space.shape}")
    print(f"  🎮 Action space: {env.action_space.nvec} actions per drone")
    print(f"  🚁 Number of drones: {config.rl_num_drones}")
    print(f"  🔧 Network architecture: [512, 512, 256, 128]")
    print(f"  🎯 Learning rate: {model.learning_rate}")
    
    return model

def create_multi_agent_callbacks(env, log_dir: str, config: SimulationConfig):
    """Create callbacks for multi-agent training"""
    
    # Evaluation callback - less frequent due to complexity
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5000,              # Less frequent evaluation
        deterministic=True,
        render=False,
        n_eval_episodes=3,           # Fewer episodes due to complexity
        verbose=1
    )
    
    # Multi-agent AoI callback
    ma_callback = MultiAgentAoICallback(log_freq=50)
    
    return CallbackList([eval_callback])

def train_multi_agent_ppo():
    """Main multi-agent PPO training function"""
    
    print("🚁🚁🚁 Starting Multi-Agent PPO Training for DTN Drone Fleet")
    print("=" * 70)
    
    config = SimulationConfig()
    
    # Create multi-agent environment
    env, log_dir, test_env = create_multi_agent_environment(config, n_envs=1)
    
    # Create multi-agent PPO model (instead of DQN)
    model = create_multi_agent_ppo_model(env, config)
    
    # Create callbacks (same as before)
    callbacks = create_multi_agent_callbacks(env, log_dir, config)
    
    print("🎯 Beginning Multi-Agent PPO Training...")
    
    try:
        total_timesteps = int(config.dqn_total_timesteps * 1.5)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="multi_agent_ppo_training",  # Changed name
            progress_bar=True
        )
        print("✅ Multi-agent PPO training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(log_dir, "final_multi_agent_ppo_model")
    model.save(final_model_path)
    print(f"💾 Final multi-agent PPO model saved to: {final_model_path}")
    
    # Test the trained model
    print("\n🧪 Testing trained multi-agent PPO model...")
    test_multi_agent_model(model, test_env, episodes=3)
    
    env.close()
    test_env.close()
    return model, log_dir

def test_multi_agent_model(model, env, episodes):
    """Test the trained multi-agent model"""
    
    print(f"\n🔬 Testing multi-agent model for {episodes} episodes...")
    
    total_rewards = []
    system_metrics = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n🚁 Episode {episode + 1}:")
        
        while True:
            # Use trained policy (deterministic)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        
        # Extract multi-agent metrics
        if 'multi_agent_aoi_metrics' in info:
            ma_metrics = info['multi_agent_aoi_metrics']
            system_metrics.append(ma_metrics['system'])
            
            print(f"  Steps: {step_count}, Total Reward: {episode_reward:.2f}")
            print(f"  System delivered: {ma_metrics['system']['total_delivered']}")
            print(f"  System delivery rate: {ma_metrics['system']['delivery_rate']:.1%}")
            print(f"  System mean AoI: {ma_metrics['system']['mean_delivery_aoi']:.1f}s")
            
            # Per-drone breakdown
            for i, drone_metrics in enumerate(ma_metrics['per_drone']):
                print(f"    Drone {i}: {drone_metrics['delivered']} delivered, "
                      f"{drone_metrics['collected']} collected")
    
    # Summary
    print(f"\n📊 Multi-Agent Test Summary ({episodes} episodes):")
    print(f"  🎯 Mean total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    if system_metrics:
        print(f"  📦 Mean system delivered: {np.mean([m['total_delivered'] for m in system_metrics]):.1f}")
        print(f"  📈 Mean system delivery rate: {np.mean([m['delivery_rate'] for m in system_metrics]):.1%}")
        print(f"  ⏱️ Mean system AoI: {np.mean([m['mean_delivery_aoi'] for m in system_metrics]):.1f}s")

def compare_multi_agent_with_baselines(config: SimulationConfig, episodes=3):
    """Compare multi-agent trained model with baseline policies"""
    
    print("\n🎲 Comparing with baseline policies...")
    
    # Test random multi-agent policy
    env = MultiAgentDTNEnvironment(config)
    
    baseline_results = {}
    
    # Random policy baseline
    print("\n🎯 Random Multi-Agent Policy:")
    random_rewards = []
    random_delivered = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            # Random action for each drone
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        random_rewards.append(episode_reward)
        
        if 'multi_agent_aoi_metrics' in info:
            delivered = info['multi_agent_aoi_metrics']['system']['total_delivered']
            random_delivered.append(delivered)
            print(f"  Episode {episode+1}: {episode_reward:.0f} reward, {delivered} delivered")
    
    baseline_results['random'] = {
        'mean_reward': np.mean(random_rewards),
        'std_reward': np.std(random_rewards),
        'mean_delivered': np.mean(random_delivered) if random_delivered else 0
    }
    
    print(f"\n📊 Random Multi-Agent Policy Results:")
    print(f"  Reward: {baseline_results['random']['mean_reward']:.0f} ± {baseline_results['random']['std_reward']:.0f}")
    print(f"  Messages delivered: {baseline_results['random']['mean_delivered']:.1f}")
    
    env.close()
    return baseline_results

if __name__ == "__main__":
    # Train the multi-agent PPO
    model, log_dir = train_multi_agent_ppo()
    
    # Compare with baselines
    config = SimulationConfig()
    compare_multi_agent_with_baselines(config, episodes=3)
    
    print(f"\n🎉 Multi-Agent PPO Training complete! Check logs at: {log_dir}")
    print("📈 View training progress: tensorboard --logdir ./logs/tensorboard_logs")