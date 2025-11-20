"""
Compare all approaches: Single Agent vs Multi-Agent vs Random Baselines
"""

import numpy as np
from stable_baselines3 import DQN, PPO  # ← ADD PPO IMPORT
from src.rl.environments.single_agent_env import DTNDroneEnvironment
from src.rl.environments.multi_agent_env import MultiAgentDTNEnvironment
from src.config.simulation_config import SimulationConfig
from src.rl.testing.test_single_agent_model import find_latest_model_path
from src.rl.testing.test_multi_agent_model import find_latest_multi_agent_model_path

def test_approach(model, env, approach_name, episodes=3):
    """Test any approach and return standardized results"""
    print(f"\n🧪 Testing {approach_name} ({episodes} episodes):")
    
    results = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(200):  # Consistent shorter episodes for fair comparison
            if model is None:  # Random policy
                action = env.action_space.sample()
            else:  # Trained policy
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Debug large rewards
            if abs(reward) > 10000:
                print(f"⚠️ Large reward: {reward} at step {step} for {approach_name}")
            
            if done or truncated:
                print(f"  Episode ended at step {step}: done={done}, truncated={truncated}")
                break
        
        # Extract metrics based on environment type
        if 'multi_agent_aoi_metrics' in info:  # Multi-agent environment
            ma_metrics = info['multi_agent_aoi_metrics']
            delivered = ma_metrics.get('system', {}).get('total_delivered', 0)
            delivery_rate = ma_metrics.get('system', {}).get('delivery_rate', 0.0)
            mean_aoi = ma_metrics.get('system', {}).get('mean_delivery_aoi', 0.0)
        elif 'aoi_metrics' in info:  # Single-agent environment
            aoi_data = info['aoi_metrics']
            delivered = aoi_data['delivered']['count']
            delivery_rate = aoi_data['global']['delivery_rate']
            mean_aoi = aoi_data['delivered']['mean_aoi']
        else:  # No metrics available
            delivered = 0
            delivery_rate = 0.0
            mean_aoi = 0.0
        
        results.append({
            'reward': episode_reward,
            'delivered': delivered,
            'delivery_rate': delivery_rate,
            'mean_aoi': mean_aoi
        })
        
        print(f"  Episode {episode+1}: {episode_reward:.0f} reward, {delivered} delivered")
    
    # Calculate statistics
    mean_reward = np.mean([r['reward'] for r in results])
    std_reward = np.std([r['reward'] for r in results])
    mean_delivered = np.mean([r['delivered'] for r in results])
    mean_delivery_rate = np.mean([r['delivery_rate'] for r in results])
    mean_aoi = np.mean([r['mean_aoi'] for r in results])
    
    print(f"  📊 {approach_name} Results:")
    print(f"     Reward: {mean_reward:.0f} ± {std_reward:.0f}")
    print(f"     Delivered: {mean_delivered:.1f}")
    print(f"     Delivery Rate: {mean_delivery_rate:.1%}")
    print(f"     Mean AoI: {mean_aoi:.1f}s")
    
    return {
        'name': approach_name,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_delivered': mean_delivered,
        'mean_delivery_rate': mean_delivery_rate,
        'mean_aoi': mean_aoi,
        'results': results
    }

def compare_all_approaches():
    """Compare all approaches: Single Agent, Multi-Agent, and baselines"""
    
    print("🏆 COMPREHENSIVE APPROACH COMPARISON")
    print("=" * 70)
    
    config = SimulationConfig()
    all_results = []
    
    # 1. Test Single Agent Trained Model
    try:
        print("\n1️⃣ Loading Single Agent Model...")
        single_model_path = find_latest_model_path("./logs/single_agent")
        single_model = DQN.load(single_model_path)
        single_env = DTNDroneEnvironment(config)
        
        single_results = test_approach(single_model, single_env, "Single Agent Trained", episodes=5)
        all_results.append(single_results)
        single_env.close()
        
    except Exception as e:
        print(f"❌ Single agent test failed: {e}")
    
    # 2. Test Multi-Agent Trained Model - FIXED VERSION
    try:
        print("\n2️⃣ Loading Multi-Agent Model...")
        multi_model_path = find_latest_multi_agent_model_path()  # ← FIXED: Remove logs_dir parameter
        multi_model = PPO.load(multi_model_path)  # ← FIXED: Use PPO instead of DQN
        multi_env = MultiAgentDTNEnvironment(config)
        
        multi_results = test_approach(multi_model, multi_env, "Multi-Agent Trained", episodes=5)
        all_results.append(multi_results)
        multi_env.close()
        
    except Exception as e:
        print(f"❌ Multi-agent test failed: {e}")
        import traceback
        traceback.print_exc()  # ← ADD: Better error debugging
    
    # 3. Test Single Agent Random Baseline
    try:
        print("\n3️⃣ Testing Single Agent Random Baseline...")
        single_random_env = DTNDroneEnvironment(config)
        
        single_random_results = test_approach(None, single_random_env, "Single Agent Random", episodes=3)
        all_results.append(single_random_results)
        single_random_env.close()
        
    except Exception as e:
        print(f"❌ Single agent random test failed: {e}")
    
    # 4. Test Multi-Agent Random Baseline
    try:
        print("\n4️⃣ Testing Multi-Agent Random Baseline...")
        multi_random_env = MultiAgentDTNEnvironment(config)
        
        multi_random_results = test_approach(None, multi_random_env, "Multi-Agent Random", episodes=3)
        all_results.append(multi_random_results)
        multi_random_env.close()
        
    except Exception as e:
        print(f"❌ Multi-agent random test failed: {e}")
    
    # Final Comparison Table
    if len(all_results) >= 2:
        print("\n" + "="*70)
        print("🏆 FINAL COMPARISON TABLE")
        print("="*70)
        
        print(f"{'Approach':<25} {'Reward':<15} {'Delivered':<12} {'Del. Rate':<10} {'Mean AoI':<10}")
        print("-" * 70)
        
        for result in all_results:
            print(f"{result['name']:<25} "
                  f"{result['mean_reward']:>7.0f}±{result['std_reward']:<6.0f} "
                  f"{result['mean_delivered']:>8.1f}    "
                  f"{result['mean_delivery_rate']:>7.1%}   "
                  f"{result['mean_aoi']:>7.1f}s")
        
        # Calculate improvements
        print("\n🚀 IMPROVEMENT ANALYSIS:")
        
        # Find baselines
        single_random = next((r for r in all_results if "Single Agent Random" in r['name']), None)
        multi_random = next((r for r in all_results if "Multi-Agent Random" in r['name']), None)
        single_trained = next((r for r in all_results if "Single Agent Trained" in r['name']), None)
        multi_trained = next((r for r in all_results if "Multi-Agent Trained" in r['name']), None)
        
        if single_trained and single_random:
            improvement = ((single_trained['mean_reward'] - single_random['mean_reward']) / 
                          abs(single_random['mean_reward']) * 100)
            print(f"   Single Agent vs Random: {improvement:.1f}% improvement")
        
        if multi_trained and multi_random:
            improvement = ((multi_trained['mean_reward'] - multi_random['mean_reward']) / 
                          abs(multi_random['mean_reward']) * 100)
            print(f"   Multi-Agent vs Random: {improvement:.1f}% improvement")
        
        if multi_trained and single_trained:
            improvement = ((multi_trained['mean_reward'] - single_trained['mean_reward']) / 
                          abs(single_trained['mean_reward']) * 100)
            print(f"   Multi-Agent vs Single Agent: {improvement:.1f}% improvement")
        
        # Comprehensive analysis
        print(f"\n📈 COMPREHENSIVE ANALYSIS:")
        
        # Best by different metrics
        best_reward = max(all_results, key=lambda x: x['mean_reward'])
        best_delivery_rate = max(all_results, key=lambda x: x['mean_delivery_rate'])
        best_delivered = max(all_results, key=lambda x: x['mean_delivered'])
        
        print(f"   🏆 Best Reward: {best_reward['name']} ({best_reward['mean_reward']:.0f})")
        print(f"   📦 Best Delivery Rate: {best_delivery_rate['name']} ({best_delivery_rate['mean_delivery_rate']:.1%})")
        print(f"   📊 Most Messages Delivered: {best_delivered['name']} ({best_delivered['mean_delivered']:.1f})")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        
        trained_approaches = [r for r in all_results if 'Trained' in r['name']]
        random_approaches = [r for r in all_results if 'Random' in r['name']]
        
        if trained_approaches and random_approaches:
            avg_trained_reward = np.mean([r['mean_reward'] for r in trained_approaches])
            avg_random_reward = np.mean([r['mean_reward'] for r in random_approaches])
            avg_trained_delivery = np.mean([r['mean_delivered'] for r in trained_approaches])
            avg_random_delivery = np.mean([r['mean_delivered'] for r in random_approaches])
            
            if avg_trained_reward > avg_random_reward:
                print(f"   ✅ Training is working: Trained models outperform random by {avg_trained_reward - avg_random_reward:.0f} reward")
            else:
                print(f"   ❌ Training issues: Random models outperform trained by {avg_random_reward - avg_trained_reward:.0f} reward")
                
            if avg_trained_delivery > avg_random_delivery:
                print(f"   ✅ Learning meaningful behavior: {avg_trained_delivery - avg_random_delivery:.1f} more deliveries")
            else:
                print(f"   ❌ Not learning effectively: {avg_random_delivery - avg_trained_delivery:.1f} fewer deliveries")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if any('❌' in line for line in [f"Training issues" if trained_approaches and random_approaches and np.mean([r['mean_reward'] for r in trained_approaches]) <= np.mean([r['mean_reward'] for r in random_approaches]) else ""]):
            print(f"   🔧 Consider adjusting reward function or training hyperparameters")
            print(f"   🔧 Check for negative AoI values or time synchronization issues")
            print(f"   🔧 Verify environment is providing meaningful learning signals")
        else:
            print(f"   🚀 Training appears successful! Consider longer training for further improvement")
            print(f"   📊 Analyze TensorBoard logs for training stability and convergence")
    
    print(f"\n🎉 Comprehensive comparison complete!")

if __name__ == "__main__":
    compare_all_approaches()