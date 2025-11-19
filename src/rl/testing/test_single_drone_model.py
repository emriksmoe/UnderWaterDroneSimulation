""""
Test the trained single drone RL model in the DTN environment.
"""

import numpy as np
from stable_baselines3 import DQN
from src.rl.environments.single_agent_env import DTNDroneEnvironment
from src.config.simulation_config import SimulationConfig

def test_single_episode(model, env):
    """Test the model for one episode and return detailed results"""
    obs, info = env.reset()
    total_reward = 0
    messages_delivered = 0
    actions_taken = []
    step_rewards = []

    print("Running single episode test...")

    for step in range(1000):  # Limit to 1000 steps for testing
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        actions_taken.append(action)
        step_rewards.append(reward)

        if step % 100 == 0:
            print(f"Step {step}: Action={action}, Reward={reward:.1f}")
        
        if done or truncated:
            if 'aoi_metrics' in info:
                aoi_data = info['aoi_metrics']
                messages_delivered = aoi_data['delivered']['count']
                delivery_rate = aoi_data['global']['delivery_rate']
                mean_aoi = aoi_data['delivered']['mean_aoi']
                
            print(f"\Episode Results:")
            print(f"   Total steps: {step + 1}")
            print(f"   Total reward: {total_reward:.2f}")
            print(f"   Messages delivered: {messages_delivered}")
            print(f"   Delivery rate: {delivery_rate:.1%}")
            print(f"   Mean AoI: {mean_aoi:.1f}s")
            
            return {
                'total_reward': total_reward,
                'steps': step + 1,
                'messages_delivered': messages_delivered,
                'delivery_rate': delivery_rate,
                'mean_aoi': mean_aoi,
                'actions_taken': actions_taken,
                'step_rewards': step_rewards
            }
    
    print(" Episode reached maximum steps without completion")
    return None

def test_multiple_episodes(model, env, num_episodes=5):
    """Test the model for multiple episodes and show statistics"""
    print(f"\nRunning {num_episodes} episode test...")
    
    results = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        result = test_single_episode(model, env)
        if result:
            results.append(result)
    
    if results:
        print(f"\nSummary Statistics ({len(results)} episodes):")
        print(f"   Mean reward: {np.mean([r['total_reward'] for r in results]):.2f} ± {np.std([r['total_reward'] for r in results]):.2f}")
        print(f"   Mean steps: {np.mean([r['steps'] for r in results]):.1f} ± {np.std([r['steps'] for r in results]):.1f}")
        print(f"   Mean messages delivered: {np.mean([r['messages_delivered'] for r in results]):.1f}")
        print(f"   Mean delivery rate: {np.mean([r['delivery_rate'] for r in results]):.1%}")
        print(f"   Mean AoI: {np.mean([r['mean_aoi'] for r in results]):.1f}s")
        
        # Action distribution analysis
        all_actions = []
        for result in results:
            all_actions.extend(result['actions_taken'])
        
        action_counts = np.bincount(all_actions, minlength=23)  # 20 sensors + 2 ships + 1 explore
        
        print(f"\nAction Distribution:")
        print(f"   Sensor visits: {sum(action_counts[:20])} ({sum(action_counts[:20])/len(all_actions)*100:.1f}%)")
        print(f"   Ship visits: {sum(action_counts[20:22])} ({sum(action_counts[20:22])/len(all_actions)*100:.1f}%)")
        print(f"   Explore actions: {action_counts[22]} ({action_counts[22]/len(all_actions)*100:.1f}%)")
        
        return results
    else:
        print("No successful episodes completed")
        return []

def compare_with_random_baseline(env, episodes=3):
    """Compare trained model with random policy"""
    print(f"\nRandom Policy Baseline ({episodes} episodes):")
    
    random_results = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        messages_delivered = 0
        
        for step in range(1000):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                if 'aoi_metrics' in info:
                    aoi_data = info['aoi_metrics']
                    messages_delivered = aoi_data['delivered']['count']
                    delivery_rate = aoi_data['global']['delivery_rate']
                
                random_results.append({
                    'total_reward': total_reward,
                    'messages_delivered': messages_delivered,
                    'delivery_rate': delivery_rate
                })
                break
    
    if random_results:
        print(f"   Mean reward: {np.mean([r['total_reward'] for r in random_results]):.2f}")
        print(f"   Mean messages delivered: {np.mean([r['messages_delivered'] for r in random_results]):.1f}")
        print(f"   Mean delivery rate: {np.mean([r['delivery_rate'] for r in random_results]):.1%}")
        
        return random_results
    else:
        print("   No baseline results collected")
        return []

def main():
    """Main testing function"""
    print("Testing Trained DQN Model for AoI Optimization")
    print("=" * 60)
    
    # Load model
    model_path = "./logs/training_20251118_211625/best_model/best_model.zip"
    
    try:
        model = DQN.load(model_path)
        print(f" Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f" Failed to load model: {e}")
        return
    
    # Setup environment
    config = SimulationConfig()
    env = DTNDroneEnvironment(config)
    
    print(f"\n Test Configuration:")
    print(f"   Area size: {config.area_size}")
    print(f"   Sensors: {config.num_sensors}")
    print(f"   Ships: {config.num_ships}")
    print(f"   Episode max steps: {config.max_episode_steps}")
    
    # Run tests
    try:
        # 1. Single episode detailed test
        print("\n" + "="*60)
        print(" SINGLE EPISODE DETAILED TEST")
        print("="*60)
        single_result = test_single_episode(model, env)
        
        # 2. Multiple episode statistics
        print("\n" + "="*60)
        print(" MULTIPLE EPISODE STATISTICS")
        print("="*60)
        trained_results = test_multiple_episodes(model, env, num_episodes=5)
        
        # 3. Random baseline comparison
        print("\n" + "="*60)
        print(" RANDOM BASELINE COMPARISON")
        print("="*60)
        random_results = compare_with_random_baseline(env, episodes=3)
        
        # 4. Final comparison
        if trained_results and random_results:
            trained_avg_reward = np.mean([r['total_reward'] for r in trained_results])
            random_avg_reward = np.mean([r['total_reward'] for r in random_results])
            improvement = trained_avg_reward - random_avg_reward
            
            print(f"\n PERFORMANCE COMPARISON:")
            print(f"   Trained Agent: {trained_avg_reward:.2f} reward")
            print(f"   Random Policy: {random_avg_reward:.2f} reward")
            print(f"   Improvement: {improvement:.2f} ({improvement/abs(random_avg_reward)*100:.1f}% better)")
            
            if improvement > 1000:
                print("   EXCELLENT: Your agent learned very effective AoI optimization!")
            elif improvement > 100:
                print("   GOOD: Your agent shows solid learning progress!")
            elif improvement > 0:
                print("   MODEST: Your agent learned something, but could be improved")
            else:
                print("   POOR: Agent performance is not better than random - check training")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    print(f"\n Testing complete!")

if __name__ == "__main__":
    main()