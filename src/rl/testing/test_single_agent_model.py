""""
Test the trained single drone RL model in the DTN environment.
"""

import numpy as np
from stable_baselines3 import DQN
from src.rl.environments.single_agent_env import DTNDroneEnvironment
from src.config.simulation_config import SimulationConfig
import os
import glob
from pathlib import Path
from visualization.rl_plots import create_test_visualization


def find_latest_model_path(logs_dir="./logs/single_agent"):
    """Find the most recent trained model automatically"""
    
    # Find all training folders
    training_folders = glob.glob(os.path.join(logs_dir, "training_*"))
    
    if not training_folders:
        raise FileNotFoundError(f"No training folders found in {logs_dir}")
    
    # Sort by folder name (which includes timestamp) - most recent first
    training_folders.sort(reverse=True)
    
    # Look for models in priority order
    model_search_paths = [
        "best_model/best_model.zip",  # Eval callback saves here
        "best_model.zip",              # Sometimes saved here
        "final_model.zip",             # Final model after training
    ]
    
    # Try each folder
    for folder in training_folders:
        print(f"🔍 Checking folder: {folder}")
        
        # Try each possible model path
        for model_file in model_search_paths:
            model_path = os.path.join(folder, model_file)
            if os.path.exists(model_path):
                print(f"✅ Found model: {model_path}")
                return model_path
        
        # List what's actually in the folder for debugging
        try:
            contents = os.listdir(folder)
            print(f"   Contents: {contents}")
        except:
            pass
    
    raise FileNotFoundError(
        f"No model found in any training folder.\n"
        f"Searched in: {logs_dir}\n"
        f"Training folders found: {len(training_folders)}\n"
        f"Run training first: python -m src.rl.training.train_single_agent"
    )


def calculate_system_aoi(env):
    """
    Calculate total system AoI for ALL messages that existed in the episode.
    
    For delivered messages: AoI = delivery_time - generation_time
    For undelivered messages: AoI = current_time - generation_time (still accumulating)
    
    This represents the TRUE system AoI - how long ALL messages have been waiting.
    """
    total_aoi = 0.0
    message_count = 0
    current_time = env.current_time
    
    # Track all message IDs we've seen to avoid double counting
    all_message_ids = set()
    
    # 1. Messages delivered to ships (AoI is FIXED at delivery time)
    delivered_aoi = 0.0
    delivered_count = 0
    for ship in env.mock_ships:
        for msg_id, delivery_time in ship.delivery_log.items():
            # Find the message to get generation time
            for msg in ship.received_messages:
                if msg.id == msg_id:
                    # For delivered messages, AoI stops accumulating at delivery
                    aoi = delivery_time - msg.generation_time
                    total_aoi += aoi
                    delivered_aoi += aoi
                    message_count += 1
                    delivered_count += 1
                    all_message_ids.add(msg_id)
                    break
    
    # 2. Messages still in sensors (AoI is CURRENT age - still accumulating)
    sensor_aoi = 0.0
    sensor_count = 0
    for sensor in env.mock_sensors:
        for msg in sensor.messages:
            if msg.id not in all_message_ids:
                # For undelivered messages, AoI continues to grow
                age = current_time - msg.generation_time
                total_aoi += age
                sensor_aoi += age
                message_count += 1
                sensor_count += 1
                all_message_ids.add(msg.id)
    
    # 3. Messages in drone buffer (AoI is CURRENT age - still accumulating)
    drone_aoi = 0.0
    drone_count = 0
    for msg in env.mock_drone.messages:
        if msg.id not in all_message_ids:
            # For undelivered messages in transit, AoI continues to grow
            age = current_time - msg.generation_time
            total_aoi += age
            drone_aoi += age
            message_count += 1
            drone_count += 1
            all_message_ids.add(msg.id)
    
    undelivered_count = sensor_count + drone_count
    
    # Calculate weighted AoI - penalize undelivered messages more
    # This represents the "true cost" where undelivered messages are assumed
    # to be delivered at episode end (worst case scenario)
    weighted_aoi = delivered_aoi
    for sensor in env.mock_sensors:
        for msg in sensor.messages:
            # Worst case: message would be delivered at episode end
            worst_case_aoi = current_time - msg.generation_time
            weighted_aoi += worst_case_aoi * 2.0  # 2x penalty for being undelivered
    
    for msg in env.mock_drone.messages:
        worst_case_aoi = current_time - msg.generation_time
        weighted_aoi += worst_case_aoi * 1.5  # 1.5x penalty (at least collected)
    
    return {
        'total_system_aoi': total_aoi,
        'message_count': message_count,
        'mean_system_aoi': total_aoi / message_count if message_count > 0 else 0.0,
        'delivered_count': delivered_count,
        'undelivered_count': undelivered_count,
        'sensor_messages': sensor_count,
        'drone_messages': drone_count,
        'mean_sensor_aoi': sensor_aoi / sensor_count if sensor_count > 0 else 0.0,
        'mean_drone_aoi': drone_aoi / drone_count if drone_count > 0 else 0.0,
        'mean_delivered_aoi': delivered_aoi / delivered_count if delivered_count > 0 else 0.0,
        'delivery_rate': (delivered_count / message_count * 100) if message_count > 0 else 0.0,
        'weighted_system_aoi': weighted_aoi / message_count if message_count > 0 else 0.0,
        'total_delivered_aoi': delivered_aoi,
        'total_sensor_aoi': sensor_aoi,
        'total_drone_aoi': drone_aoi
    }


def test_single_episode(model, env, verbose=True):
    """Test the model for one episode and return detailed results"""
    obs, info = env.reset()
    total_reward = 0
    messages_delivered = 0
    delivery_rate = 0.0
    mean_aoi = 0.0
    min_aoi = 0.0
    max_aoi = 0.0
    actions_taken = []
    step_rewards = []
    
    # ✅ Track what actions are being taken
    action_counts = {
        'sensor': 0,
        'ship': 0,
        'explore': 0
    }
    
    # ✅ Store step-by-step logs
    step_logs = []

    if verbose:
        print("Running single episode test...")

    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        
        # ✅ Track action type
        if action < 20:
            action_counts['sensor'] += 1
        elif action < 22:
            action_counts['ship'] += 1
        else:
            action_counts['explore'] += 1
        
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        actions_taken.append(action)
        step_rewards.append(reward)

        # ✅ Store log for later display
        if step % 50 == 0:
            log_entry = (
                f"Step {step}: Action={action}, Reward={reward:.1f}, Total Reward={total_reward:.1f}\n"
                f"   Action distribution so far: Sensors={action_counts['sensor']}, Ships={action_counts['ship']}, Explore={action_counts['explore']}"
            )
            step_logs.append(log_entry)
            if verbose:
                print(log_entry)
        
        if done or truncated:
            # ✅ Store final episode info
            episode_stats = info.get('stats', {})
            
            final_log = [
                "\n🔍 Episode Stats from info:",
                f"   Messages collected: {episode_stats.get('messages_collected', 0)}",
                f"   Messages delivered: {episode_stats.get('messages_delivered', 0)}",
                f"   Sensor visits: {episode_stats.get('sensor_visits', 0)}",
                f"   Ship visits: {episode_stats.get('ship_visits', 0)}"
            ]
            
            # Extract metrics from the FINAL info dict
            if 'aoi_metrics' in info:
                aoi_data = info['aoi_metrics']
                delivered_data = aoi_data.get('delivered', {})
                global_data = aoi_data.get('global', {})
                
                messages_delivered = delivered_data.get('count', 0)
                delivery_rate = global_data.get('delivery_rate', 0.0)
                mean_aoi = delivered_data.get('mean_aoi', 0.0)
                min_aoi = delivered_data.get('min_aoi', 0.0)
                max_aoi = delivered_data.get('max_aoi', 0.0)
                
                final_log.extend([
                    "\n🔍 Debug - AoI Metrics Extracted:",
                    f"   Delivered count: {messages_delivered}",
                    f"   Delivery rate: {delivery_rate:.1%}",
                    f"   Mean AoI: {mean_aoi:.1f}s",
                    f"   Min AoI: {min_aoi:.1f}s",
                    f"   Max AoI: {max_aoi:.1f}s"
                ])
                
                # ✅ Check if there's a mismatch
                if messages_delivered != episode_stats.get('messages_delivered', 0):
                    final_log.extend([
                        "\n⚠️ MISMATCH DETECTED!",
                        f"   aoi_metrics delivered: {messages_delivered}",
                        f"   episode_stats delivered: {episode_stats.get('messages_delivered', 0)}"
                    ])
            else:
                final_log.extend([
                    "\n⚠️ Warning: No aoi_metrics in final info!",
                    f"   Available info keys: {info.keys()}"
                ])
            
            # Calculate system AoI at episode end
            system_aoi = calculate_system_aoi(env)
            
            final_log.extend([
                "\n📊 Episode Results:",
                f"   Total steps: {step + 1}",
                f"   Total reward: {total_reward:.2f}",
                f"   Messages delivered: {messages_delivered}",
                f"   Delivery rate: {delivery_rate:.1%}",
                f"   Mean delivered AoI: {mean_aoi:.1f}s",
                f"   Min/Max delivered AoI: {min_aoi:.1f}s / {max_aoi:.1f}s",
                "\n🌐 System-Wide AoI (ALL messages that existed in episode):",
                f"   Total messages: {system_aoi['message_count']}",
                f"   Delivered: {system_aoi['delivered_count']} (AoI fixed at delivery)",
                f"   Undelivered: {system_aoi['undelivered_count']} (AoI still growing)",
                f"      ├─ In sensors: {system_aoi['sensor_messages']}",
                f"      └─ In drone: {system_aoi['drone_messages']}",
                f"\n   Mean System AoI (all messages): {system_aoi['mean_system_aoi']:.1f}s",
                f"      ├─ Delivered avg: {system_aoi['mean_delivered_aoi']:.1f}s (fixed)",
                f"      ├─ Sensor avg: {system_aoi['mean_sensor_aoi']:.1f}s (growing)",
                f"      └─ Drone avg: {system_aoi['mean_drone_aoi']:.1f}s (growing)",
                f"\n   Total AoI accumulation:",
                f"      ├─ Delivered: {system_aoi['total_delivered_aoi']:.1f}s",
                f"      ├─ Sensors: {system_aoi['total_sensor_aoi']:.1f}s",
                f"      └─ Drone: {system_aoi['total_drone_aoi']:.1f}s",
                f"      = Total: {system_aoi['total_system_aoi']:.1f}s",
                f"\n   Weighted System AoI (penalizes undelivered): {system_aoi['weighted_system_aoi']:.1f}s",
                "\n🎯 Final Action Distribution:",
                f"   Sensors: {action_counts['sensor']} ({action_counts['sensor']/(step+1)*100:.1f}%)",
                f"   Ships: {action_counts['ship']} ({action_counts['ship']/(step+1)*100:.1f}%)",
                f"   Explore: {action_counts['explore']} ({action_counts['explore']/(step+1)*100:.1f}%)"
            ])
            
            # Print at episode end
            if verbose:
                for log in final_log:
                    print(log)
            
            return {
                'total_reward': total_reward,
                'steps': step + 1,
                'messages_delivered': messages_delivered,
                'delivery_rate': delivery_rate,
                'mean_aoi': mean_aoi,
                'min_aoi': min_aoi,
                'max_aoi': max_aoi,
                'system_aoi': system_aoi,
                'actions_taken': actions_taken,
                'step_rewards': step_rewards,
                'step_logs': step_logs,
                'final_log': final_log,
                'episode_stats': episode_stats,
                'action_counts': action_counts
            }
    
    if verbose:
        print("Episode reached maximum steps without completion")
    
    # Calculate system AoI even if episode didn't finish
    system_aoi = calculate_system_aoi(env)
    
    return {
        'total_reward': total_reward,
        'steps': 1000,
        'messages_delivered': messages_delivered,
        'delivery_rate': delivery_rate,
        'mean_aoi': mean_aoi,
        'min_aoi': min_aoi,
        'max_aoi': max_aoi,
        'system_aoi': system_aoi,
        'actions_taken': actions_taken,
        'step_rewards': step_rewards,
        'step_logs': step_logs,
        'final_log': [],
        'episode_stats': {},
        'action_counts': action_counts
    }

def test_multiple_episodes(model, env, num_episodes=5):
    """Test the model for multiple episodes and show statistics"""
    print(f"\nRunning {num_episodes} episode test...")
    
    results = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        result = test_single_episode(model, env, verbose=False)  # Don't print during multi-episode
        if result:
            results.append(result)
            # Print summary for this episode
            print(f"Episode {episode + 1} Summary:")
            print(f"   Reward: {result['total_reward']:.2f}")
            print(f"   Messages delivered: {result['messages_delivered']}")
            print(f"   Delivery rate: {result['delivery_rate']:.1%}")
            print(f"   Mean delivered AoI: {result['mean_aoi']:.1f}s")
            if 'system_aoi' in result:
                print(f"   Mean system AoI (all msgs): {result['system_aoi']['mean_system_aoi']:.1f}s")
                print(f"   Weighted system AoI: {result['system_aoi']['weighted_system_aoi']:.1f}s")
    
    if results:
        print(f"\n{'='*60}")
        print(f"📊 Summary Statistics ({len(results)} episodes)")
        print(f"{'='*60}")
        
        # Reward statistics
        rewards = [r['total_reward'] for r in results]
        print(f"\n💰 Reward Metrics:")
        print(f"   Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"   Min reward: {np.min(rewards):.2f}")
        print(f"   Max reward: {np.max(rewards):.2f}")
        
        # Step statistics
        steps = [r['steps'] for r in results]
        print(f"\n📏 Episode Length:")
        print(f"   Mean steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"   Min/Max steps: {np.min(steps)}/{np.max(steps)}")
        
        # Delivery statistics
        msgs_delivered = [r['messages_delivered'] for r in results]
        delivery_rates = [r['delivery_rate'] for r in results]
        print(f"\n📨 Delivery Metrics:")
        print(f"   Mean messages delivered: {np.mean(msgs_delivered):.1f} ± {np.std(msgs_delivered):.1f}")
        print(f"   Mean delivery rate: {np.mean(delivery_rates):.1%} ± {np.std(delivery_rates):.1%}")
        print(f"   Min/Max deliveries: {np.min(msgs_delivered)}/{np.max(msgs_delivered)}")
        
        # AoI statistics (delivered messages only)
        aoi_values = [r['mean_aoi'] for r in results if r['mean_aoi'] > 0]
        if aoi_values:
            print(f"\n⏱️  AoI Metrics (Delivered Messages):")
            print(f"   Mean AoI: {np.mean(aoi_values):.1f}s ± {np.std(aoi_values):.1f}s")
            print(f"   Min AoI: {np.min(aoi_values):.1f}s")
            print(f"   Max AoI: {np.max(aoi_values):.1f}s")
            
            # Min/Max AoI across all deliveries
            all_min_aois = [r['min_aoi'] for r in results if r['min_aoi'] > 0]
            all_max_aois = [r['max_aoi'] for r in results if r['max_aoi'] > 0]
            if all_min_aois and all_max_aois:
                print(f"   Best delivery AoI: {np.min(all_min_aois):.1f}s")
                print(f"   Worst delivery AoI: {np.max(all_max_aois):.1f}s")
        else:
            print(f"\n⏱️  AoI Metrics (Delivered): N/A (no deliveries)")
        
        # System-wide AoI statistics
        system_aois = [r['system_aoi']['mean_system_aoi'] for r in results if 'system_aoi' in r]
        weighted_aois = [r['system_aoi']['weighted_system_aoi'] for r in results if 'system_aoi' in r]
        
        if system_aois:
            print(f"\n🌐 System-Wide AoI Metrics (All Messages That Existed):")
            print(f"   Mean system AoI: {np.mean(system_aois):.1f}s ± {np.std(system_aois):.1f}s")
            print(f"   Min/Max system AoI: {np.min(system_aois):.1f}s / {np.max(system_aois):.1f}s")
            
            if weighted_aois:
                print(f"\n   Weighted system AoI (penalizes undelivered):")
                print(f"   Mean: {np.mean(weighted_aois):.1f}s ± {np.std(weighted_aois):.1f}s")
            
            # Average undelivered count
            undelivered_counts = [r['system_aoi']['undelivered_count'] for r in results if 'system_aoi' in r]
            total_message_counts = [r['system_aoi']['message_count'] for r in results if 'system_aoi' in r]
            print(f"\n   Mean undelivered messages: {np.mean(undelivered_counts):.1f} / {np.mean(total_message_counts):.1f} total")
        
        # Action distribution analysis
        all_actions = []
        for result in results:
            all_actions.extend(result['actions_taken'])
        
        if all_actions:
            action_counts = np.bincount(all_actions, minlength=23)  # 20 sensors + 2 ships + 1 explore
            
            print(f"\n🎯 Action Distribution:")
            print(f"   Total actions: {len(all_actions)}")
            print(f"   Sensor visits: {sum(action_counts[:20])} ({sum(action_counts[:20])/len(all_actions)*100:.1f}%)")
            print(f"   Ship visits: {sum(action_counts[20:22])} ({sum(action_counts[20:22])/len(all_actions)*100:.1f}%)")
            print(f"   Explore actions: {action_counts[22]} ({action_counts[22]/len(all_actions)*100:.1f}%)")
            
            # Most visited entities
            print(f"\n   Most visited sensors:")
            sensor_visits = action_counts[:20]
            top_sensors = np.argsort(sensor_visits)[-3:][::-1]
            for idx in top_sensors:
                if sensor_visits[idx] > 0:
                    print(f"      Sensor {idx}: {sensor_visits[idx]} visits ({sensor_visits[idx]/len(all_actions)*100:.1f}%)")
        
        return results
    else:
        print("No successful episodes completed")
        return []

def compare_with_random_baseline(env, episodes=3):
    """Compare trained model with random policy"""
    print(f"\n{'='*60}")
    print(f"🎲 Random Policy Baseline ({episodes} episodes)")
    print(f"{'='*60}")
    
    random_results = []
    
    for episode in range(episodes):
        print(f"\nRandom Episode {episode + 1}/{episodes}")
        obs, info = env.reset()
        total_reward = 0
        messages_delivered = 0
        delivery_rate = 0.0
        mean_aoi = 0.0
        min_aoi = 0.0
        max_aoi = 0.0
        
        for step in range(1000):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                # Extract metrics
                if 'aoi_metrics' in info:
                    aoi_data = info['aoi_metrics']
                    delivered_data = aoi_data.get('delivered', {})
                    global_data = aoi_data.get('global', {})
                    
                    messages_delivered = delivered_data.get('count', 0)
                    delivery_rate = global_data.get('delivery_rate', 0.0)
                    mean_aoi = delivered_data.get('mean_aoi', 0.0)
                    min_aoi = delivered_data.get('min_aoi', 0.0)
                    max_aoi = delivered_data.get('max_aoi', 0.0)
                
                # Calculate system AoI
                system_aoi = calculate_system_aoi(env)
                
                random_results.append({
                    'total_reward': total_reward,
                    'messages_delivered': messages_delivered,
                    'delivery_rate': delivery_rate,
                    'mean_aoi': mean_aoi,
                    'min_aoi': min_aoi,
                    'max_aoi': max_aoi,
                    'system_aoi': system_aoi
                })
                
                print(f"   Reward: {total_reward:.2f}, Deliveries: {messages_delivered}, Rate: {delivery_rate:.1%}")
                print(f"   Mean delivered AoI: {mean_aoi:.1f}s")
                print(f"   Mean system AoI: {system_aoi['mean_system_aoi']:.1f}s")
                print(f"   Weighted system AoI: {system_aoi['weighted_system_aoi']:.1f}s")
                break
    
    if random_results:
        print(f"\n📊 Random Policy Summary:")
        print(f"   Mean reward: {np.mean([r['total_reward'] for r in random_results]):.2f}")
        print(f"   Mean messages delivered: {np.mean([r['messages_delivered'] for r in random_results]):.1f}")
        print(f"   Mean delivery rate: {np.mean([r['delivery_rate'] for r in random_results]):.1%}")
        
        aoi_values = [r['mean_aoi'] for r in random_results if r['mean_aoi'] > 0]
        if aoi_values:
            print(f"   Mean delivered AoI: {np.mean(aoi_values):.1f}s")
        
        system_aois = [r['system_aoi']['mean_system_aoi'] for r in random_results]
        weighted_aois = [r['system_aoi']['weighted_system_aoi'] for r in random_results]
        if system_aois:
            print(f"   Mean system AoI: {np.mean(system_aois):.1f}s")
            print(f"   Mean weighted system AoI: {np.mean(weighted_aois):.1f}s")
        
        return random_results
    else:
        print("   No baseline results collected")
        return []

def main():
    """Main testing function"""
    print("\n" + "="*60)
    print("🚁 Testing Trained DQN Model for AoI Optimization")
    print("="*60)
    
    # Load model
    try:
        model_path = find_latest_model_path()
        print(f"\n🔍 Auto-detected latest model:")
        print(f"   {model_path}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nMake sure you've run training first:")
        print("   python -m src.rl.training.train_single_agent")
        return
    
    try:
        model = DQN.load(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Setup environment
    config = SimulationConfig()
    env = DTNDroneEnvironment(config)
    
    print(f"\n📋 Test Configuration:")
    print(f"   Area size: {config.area_size}")
    print(f"   Sensors: {config.num_sensors}")
    print(f"   Ships: {config.num_ships}")
    print(f"   Episode max steps: {config.max_episode_steps}")
    
    # Run tests
    try:
        # 1. Single episode detailed test
        print("\n" + "="*60)
        print("📊 SINGLE EPISODE DETAILED TEST")
        print("="*60)
        single_result = test_single_episode(model, env, verbose=True)
        
        # 2. Multiple episode statistics
        print("\n" + "="*60)
        print("📈 MULTIPLE EPISODE STATISTICS")
        print("="*60)
        trained_results = test_multiple_episodes(model, env, num_episodes=5)
        
        # 3. Random baseline comparison
        print("\n" + "="*60)
        print("🎲 RANDOM BASELINE COMPARISON")
        print("="*60)
        random_results = compare_with_random_baseline(env, episodes=3)
        
        # 4. Final comparison and visualization
        if trained_results and random_results:
            # Calculate averages
            trained_avg_reward = np.mean([r['total_reward'] for r in trained_results])
            random_avg_reward = np.mean([r['total_reward'] for r in random_results])
            
            trained_msgs = np.mean([r['messages_delivered'] for r in trained_results])
            random_msgs = np.mean([r['messages_delivered'] for r in random_results])
            trained_delivery = np.mean([r['delivery_rate'] for r in trained_results])
            random_delivery = np.mean([r['delivery_rate'] for r in random_results])
            
            trained_aois = [r['mean_aoi'] for r in trained_results if r['mean_aoi'] > 0]
            random_aois = [r['mean_aoi'] for r in random_results if r['mean_aoi'] > 0]
            trained_mean_aoi = np.mean(trained_aois) if trained_aois else 0.0
            random_mean_aoi = np.mean(random_aois) if random_aois else 0.0
            
            trained_system_aois = [r['system_aoi']['mean_system_aoi'] for r in trained_results if 'system_aoi' in r]
            random_system_aois = [r['system_aoi']['mean_system_aoi'] for r in random_results if 'system_aoi' in r]
            trained_mean_system_aoi = np.mean(trained_system_aois) if trained_system_aois else 0.0
            random_mean_system_aoi = np.mean(random_system_aois) if random_system_aois else 0.0
            
            trained_weighted_aois = [r['system_aoi']['weighted_system_aoi'] for r in trained_results if 'system_aoi' in r]
            random_weighted_aois = [r['system_aoi']['weighted_system_aoi'] for r in random_results if 'system_aoi' in r]
            trained_mean_weighted_aoi = np.mean(trained_weighted_aois) if trained_weighted_aois else 0.0
            random_mean_weighted_aoi = np.mean(random_weighted_aois) if random_weighted_aois else 0.0
            
            trained_undelivered = np.mean([r['system_aoi']['undelivered_count'] for r in trained_results if 'system_aoi' in r])
            random_undelivered = np.mean([r['system_aoi']['undelivered_count'] for r in random_results if 'system_aoi' in r])
            
            # Prepare statistics for visualization
            trained_stats = {
                'mean_reward': trained_avg_reward,
                'messages_delivered': trained_msgs,
                'delivery_rate': trained_delivery,
                'mean_delivered_aoi': trained_mean_aoi,
                'mean_system_aoi': trained_mean_system_aoi,
                'undelivered_messages': trained_undelivered,
                'sensor_aoi_avg': np.mean([r['system_aoi']['mean_sensor_aoi'] for r in trained_results if 'system_aoi' in r]),
                'drone_aoi_avg': np.mean([r['system_aoi']['mean_drone_aoi'] for r in trained_results if 'system_aoi' in r])
            }
            
            random_stats = {
                'mean_reward': random_avg_reward,
                'messages_delivered': random_msgs,
                'delivery_rate': random_delivery,
                'mean_delivered_aoi': random_mean_aoi,
                'mean_system_aoi': random_mean_system_aoi,
                'undelivered_messages': random_undelivered,
                'sensor_aoi_avg': np.mean([r['system_aoi']['mean_sensor_aoi'] for r in random_results if 'system_aoi' in r]),
                'drone_aoi_avg': np.mean([r['system_aoi']['mean_drone_aoi'] for r in random_results if 'system_aoi' in r])
            }
            
            # Print text comparison (your existing code)
            print(f"\n{'='*60}")
            print(f"🏆 PERFORMANCE COMPARISON")
            print(f"{'='*60}")
            print(f"\n💰 Reward Comparison:")
            print(f"   Trained Agent: {trained_avg_reward:.2f}")
            print(f"   Random Policy: {random_avg_reward:.2f}")
            
            if random_avg_reward != 0:
                improvement_absolute = trained_avg_reward - random_avg_reward
                percent_improvement = (improvement_absolute / abs(random_avg_reward)) * 100
                print(f"   Improvement: {improvement_absolute:.2f} ({percent_improvement:.1f}% better)")
                
                if percent_improvement > 50:
                    print("\n   ✅ GOOD: Agent shows solid improvement!")
                elif percent_improvement > 10:
                    print("\n   ⚠️ MODEST: Some improvement shown")
                else:
                    print("\n   ❌ POOR: Minimal improvement over random")
            
            print(f"\n📨 Delivery Metrics Comparison:")
            print(f"   Messages Delivered:")
            print(f"      Trained: {trained_msgs:.1f} messages")
            print(f"      Random:  {random_msgs:.1f} messages")
            print(f"   Delivery Rate:")
            print(f"      Trained: {trained_delivery:.1%}")
            print(f"      Random:  {random_delivery:.1%}")
            
            if trained_msgs > random_msgs:
                improvement = ((trained_msgs - random_msgs) / random_msgs) * 100
                print(f"   ✅ Trained agent delivers {improvement:.1f}% MORE messages!")
            else:
                decrease = ((random_msgs - trained_msgs) / random_msgs) * 100
                print(f"   ≈ Similar delivery performance ({decrease:.1f}% difference)")
            
            print(f"\n⏱️  Delivered Messages AoI Comparison:")
            print(f"   Mean Delivered AoI:")
            print(f"      Trained: {trained_mean_aoi:.1f}s")
            print(f"      Random:  {random_mean_aoi:.1f}s")
            
            if trained_mean_aoi > 0 and random_mean_aoi > 0:
                aoi_improvement = ((random_mean_aoi - trained_mean_aoi) / random_mean_aoi) * 100
                if aoi_improvement > 0:
                    print(f"   ✅ Trained agent reduces delivered AoI by {aoi_improvement:.1f}% (lower is better!)")
            
            print(f"\n🌐 System-Wide AoI Comparison (ALL Messages That Existed):")
            print(f"   This is the TRUE optimization metric!")
            print(f"   (Delivered messages: AoI fixed at delivery)")
            print(f"   (Undelivered messages: AoI still accumulating)")
            print(f"\n   Mean System AoI (all messages):")
            print(f"      Trained: {trained_mean_system_aoi:.1f}s")
            print(f"      Random:  {random_mean_system_aoi:.1f}s")
            
            if trained_mean_system_aoi > 0 and random_mean_system_aoi > 0:
                system_aoi_improvement = ((random_mean_system_aoi - trained_mean_system_aoi) / random_mean_system_aoi) * 100
                if system_aoi_improvement > 0:
                    print(f"   ✅ Trained agent reduces TOTAL SYSTEM AoI by {system_aoi_improvement:.1f}%!")
                    print(f"      🎯 This means the agent IS optimizing AoI correctly!")
            
            print(f"\n   Weighted System AoI (2x penalty for undelivered):")
            print(f"      Trained: {trained_mean_weighted_aoi:.1f}s")
            print(f"      Random:  {random_mean_weighted_aoi:.1f}s")
            
            if trained_mean_weighted_aoi > 0 and random_mean_weighted_aoi > 0:
                weighted_improvement = ((random_mean_weighted_aoi - trained_mean_weighted_aoi) / random_mean_weighted_aoi) * 100
                print(f"   ✅ Weighted improvement: {weighted_improvement:.1f}%")
            
            print(f"\n   Undelivered Messages at Episode End:")
            print(f"      Trained: {trained_undelivered:.1f} messages")
            print(f"      Random:  {random_undelivered:.1f} messages")
            
            if trained_undelivered < random_undelivered:
                reduction = ((random_undelivered - trained_undelivered) / random_undelivered) * 100
                print(f"      ✅ Trained has {reduction:.1f}% fewer undelivered messages")
            
            # Generate visualizations
            print("\n" + "="*60)
            print("📊 Generating visualizations...")
            print("="*60)
            
            try:
                create_test_visualization(trained_stats, random_stats)
            except Exception as e:
                print(f"⚠️ Visualization failed: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Testing complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()