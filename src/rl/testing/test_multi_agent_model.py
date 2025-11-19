"""
Test the trained multi-agent RL model in the DTN environment.
"""

import numpy as np
from stable_baselines3 import PPO
from src.rl.environments.multi_agent_env import MultiAgentDTNEnvironment
from src.config.simulation_config import SimulationConfig
import os
import glob

def find_latest_multi_agent_model_path(logs_dir="./logs/multi_agent", model_name="best_model.zip"):
    """Find the most recent multi-agent trained model automatically"""
    
    # Find all multi-agent training folders
    training_folders = glob.glob(os.path.join(logs_dir, "training_*"))
    
    if not training_folders:
        raise FileNotFoundError(f"No multi-agent training folders found in {logs_dir}")
    
    # Sort by folder name (timestamp) - most recent first
    training_folders.sort(reverse=True)
    
    # Look for the model in the latest folder
    for folder in training_folders:
        model_path = os.path.join(folder, "best_model", model_name)
        if os.path.exists(model_path):
            return model_path
    
    raise FileNotFoundError(f"No {model_name} found in any multi-agent training folder")

def test_multi_agent_single_episode(model, env):
    """Test the multi-agent model for one episode and return detailed results"""
    obs, info = env.reset()
    total_reward = 0
    actions_taken = []
    step_rewards = []

    print("Running multi-agent single episode test...")

    for step in range(1000):  # Limit to 1000 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        actions_taken.append(action.tolist() if hasattr(action, 'tolist') else list(action))
        step_rewards.append(reward)

        if step % 100 == 0:
            print(f"Step {step}: Actions={action}, Reward={reward:.1f}")
        
        if done or truncated:
            if 'multi_agent_aoi_metrics' in info:
                ma_metrics = info['multi_agent_aoi_metrics']
                system_metrics = ma_metrics['system']
                
                print(f"\n🚁 Multi-Agent Episode Results:")
                print(f"   Total steps: {step + 1}")
                print(f"   Total reward: {total_reward:.2f}")
                print(f"   System messages delivered: {system_metrics['total_delivered']}")
                print(f"   System delivery rate: {system_metrics['delivery_rate']:.1%}")
                print(f"   System mean AoI: {system_metrics['mean_delivery_aoi']:.1f}s")
                
                # Per-drone breakdown
                print(f"\n   Per-Drone Performance:")
                for i, drone_metrics in enumerate(ma_metrics['per_drone']):
                    print(f"     Drone {i}: {drone_metrics['delivered']} delivered, "
                          f"{drone_metrics['collected']} collected")
                
                return {
                    'total_reward': total_reward,
                    'steps': step + 1,
                    'system_delivered': system_metrics['total_delivered'],
                    'system_delivery_rate': system_metrics['delivery_rate'],
                    'system_mean_aoi': system_metrics['mean_delivery_aoi'],
                    'per_drone_metrics': ma_metrics['per_drone'],
                    'actions_taken': actions_taken,
                    'step_rewards': step_rewards
                }
    
    print("🚁 Episode reached maximum steps without completion")
    return None

def test_multi_agent_multiple_episodes(model, env, num_episodes=5):
    """Test the multi-agent model for multiple episodes and show statistics"""
    print(f"\n🚁 Running {num_episodes} multi-agent episode test...")
    
    results = []
    
    for episode in range(num_episodes):
        print(f"\n--- Multi-Agent Episode {episode + 1}/{num_episodes} ---")
        result = test_multi_agent_single_episode(model, env)
        if result:
            results.append(result)
    
    if results:
        print(f"\n📊 Multi-Agent Summary Statistics ({len(results)} episodes):")
        print(f"   Mean reward: {np.mean([r['total_reward'] for r in results]):.2f} ± {np.std([r['total_reward'] for r in results]):.2f}")
        print(f"   Mean steps: {np.mean([r['steps'] for r in results]):.1f} ± {np.std([r['steps'] for r in results]):.1f}")
        print(f"   Mean system delivered: {np.mean([r['system_delivered'] for r in results]):.1f}")
        print(f"   Mean system delivery rate: {np.mean([r['system_delivery_rate'] for r in results]):.1%}")
        print(f"   Mean system AoI: {np.mean([r['system_mean_aoi'] for r in results]):.1f}s")
        
        # Multi-agent coordination analysis
        total_drone_deliveries = []
        total_drone_collections = []
        for result in results:
            episode_deliveries = [drone['delivered'] for drone in result['per_drone_metrics']]
            episode_collections = [drone['collected'] for drone in result['per_drone_metrics']]
            total_drone_deliveries.extend(episode_deliveries)
            total_drone_collections.extend(episode_collections)
        
        print(f"\n   🤝 Multi-Agent Coordination Analysis:")
        print(f"     Total fleet actions analyzed: {len([act for r in results for act in r['actions_taken']])}")
        print(f"     Mean deliveries per drone: {np.mean(total_drone_deliveries):.1f}")
        print(f"     Mean collections per drone: {np.mean(total_drone_collections):.1f}")
        print(f"     Drone delivery balance (std): {np.std(total_drone_deliveries):.1f}")
        print(f"     Drone collection balance (std): {np.std(total_drone_collections):.1f}")
        
        # Action distribution analysis for multi-agent
        all_actions = []
        for result in results:
            for action_set in result['actions_taken']:
                all_actions.extend(action_set)  # Flatten multi-drone actions
        
        if all_actions:
            action_counts = np.bincount(all_actions, minlength=23)  # 20 sensors + 2 ships + 1 explore
            
            print(f"\n   📊 Fleet Action Distribution:")
            print(f"     Sensor visits: {sum(action_counts[:20])} ({sum(action_counts[:20])/len(all_actions)*100:.1f}%)")
            print(f"     Ship visits: {sum(action_counts[20:22])} ({sum(action_counts[20:22])/len(all_actions)*100:.1f}%)")
            print(f"     Explore actions: {action_counts[22]} ({action_counts[22]/len(all_actions)*100:.1f}%)")
        
        return results
    else:
        print("❌ No successful multi-agent episodes completed")
        return []

def compare_multi_agent_with_random_baseline(env, episodes=3):
    """Compare trained multi-agent model with random multi-agent policy"""
    print(f"\n🎲 Random Multi-Agent Policy Baseline ({episodes} episodes):")
    
    random_results = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(1000):
            action = env.action_space.sample()  # Random actions for all drones
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                if 'multi_agent_aoi_metrics' in info:
                    ma_metrics = info['multi_agent_aoi_metrics']
                    system_metrics = ma_metrics['system']
                    
                    random_results.append({
                        'total_reward': total_reward,
                        'system_delivered': system_metrics['total_delivered'],
                        'system_delivery_rate': system_metrics['delivery_rate'],
                        'system_mean_aoi': system_metrics['mean_delivery_aoi']
                    })
                break
    
    if random_results:
        print(f"   Mean reward: {np.mean([r['total_reward'] for r in random_results]):.2f}")
        print(f"   Mean system delivered: {np.mean([r['system_delivered'] for r in random_results]):.1f}")
        print(f"   Mean system delivery rate: {np.mean([r['system_delivery_rate'] for r in random_results]):.1%}")
        print(f"   Mean system AoI: {np.mean([r['system_mean_aoi'] for r in random_results]):.1f}s")
        
        return random_results
    else:
        print("   ❌ No baseline results collected")
        return []

def main():
    """Main multi-agent testing function"""
    print("🚁🚁🚁 Testing Trained Multi-Agent PPO Model for AoI Optimization")
    print("=" * 70)
    
    # Load multi-agent model
    try:
        model_path = find_latest_multi_agent_model_path()
        print(f"🔍 Auto-detected latest multi-agent model: {model_path}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Make sure you've run multi-agent training first:")
        print("   python -m src.rl.training.train_multi_agent")
        return
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Multi-agent PPO model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load multi-agent model: {e}")
        return
    
    # Setup multi-agent environment
    config = SimulationConfig()
    env = MultiAgentDTNEnvironment(config)
    
    print(f"\n📋 Multi-Agent Test Configuration:")
    print(f"   🌊 Area size: {config.area_size}")
    print(f"   📡 Sensors: {config.num_sensors}")
    print(f"   🚢 Ships: {config.num_ships}")
    print(f"   🚁 Drones: {config.rl_num_drones}")
    print(f"   ⏱️  Episode max steps: {config.max_episode_steps}")
    
    # Run multi-agent tests
    try:
        # 1. Single episode detailed test
        print("\n" + "="*70)
        print("🚁 MULTI-AGENT SINGLE EPISODE DETAILED TEST")
        print("="*70)
        single_result = test_multi_agent_single_episode(model, env)
        
        # 2. Multiple episode statistics
        print("\n" + "="*70)
        print("🚁 MULTI-AGENT MULTIPLE EPISODE STATISTICS")
        print("="*70)
        trained_results = test_multi_agent_multiple_episodes(model, env, num_episodes=5)
        
        # 3. Random baseline comparison
        print("\n" + "="*70)
        print("🎲 MULTI-AGENT RANDOM BASELINE COMPARISON")
        print("="*70)
        random_results = compare_multi_agent_with_random_baseline(env, episodes=3)
        
        # 4. Final comparison
        if trained_results and random_results:
            trained_avg_reward = np.mean([r['total_reward'] for r in trained_results])
            random_avg_reward = np.mean([r['total_reward'] for r in random_results])
            improvement = trained_avg_reward - random_avg_reward
            
            print(f"\n🏆 MULTI-AGENT PERFORMANCE COMPARISON:")
            print(f"   🧠 Trained Multi-Agent: {trained_avg_reward:.2f} reward")
            print(f"   🎲 Random Multi-Agent: {random_avg_reward:.2f} reward")
            print(f"   📈 Improvement: {improvement:.2f} ({improvement/abs(random_avg_reward)*100:.1f}% better)")
            
            if improvement > 5000:
                print("   🏆 EXCELLENT: Multi-agent coordination learned very effective fleet AoI optimization!")
            elif improvement > 1000:
                print("   ✅ GOOD: Multi-agent system shows solid coordination learning!")
            elif improvement > 0:
                print("   📊 MODEST: Multi-agent learned some coordination, but could be improved")
            else:
                print("   ❌ POOR: Multi-agent performance not better than random - check training")
        
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    print(f"\n🎉 Multi-agent testing complete!")

if __name__ == "__main__":
    main()