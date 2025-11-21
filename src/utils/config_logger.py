"""Utility to log configuration parameters for experiment tracking"""

import json
from pathlib import Path
from datetime import datetime
from src.config.simulation_config import SimulationConfig


def get_model_name_from_path(model_path: str) -> str:
    """Extract model name from path (parent directory name)"""
    path = Path(model_path)
    # Model path is like: logs/single_agent/baseline_v1_20251121_143022/best_model/best_model.zip
    # We want: baseline_v1_20251121_143022
    parent_dirs = path.parts
    
    # Find the model directory (parent of best_model or final_model)
    for i, part in enumerate(parent_dirs):
        if part == 'single_agent' and i + 1 < len(parent_dirs):
            return parent_dirs[i + 1]
    
    # Fallback to parent directory name
    return path.parent.parent.name


def extract_rl_config(config: SimulationConfig) -> dict:
    """Extract only RL-related parameters from config"""
    
    rl_params = {
        # Reward parameters
        'reward_collection_base': config.reward_collection_base,
        'reward_collection_urgency_multiplier': config.reward_collection_urgency_multiplier,
        'reward_delivery_base': config.reward_delivery_base,
        'reward_delivery_freshness_multiplier': config.reward_delivery_freshness_multiplier,
        
        # Penalty parameters
        'penalty_time_per_second': config.penalty_time_per_second,
        'penalty_carrying_per_age_unit': config.penalty_carrying_per_age_unit,
        'penalty_undelivered_base': config.penalty_undelivered_base,
        'penalty_undelivered_age_multiplier': config.penalty_undelivered_age_multiplier,
        'penalty_uncollected_multiplier': config.penalty_uncollected_multiplier,
        
        # Action penalties
        'penalty_empty_sensor': config.penalty_empty_sensor,
        'penalty_ship_no_messages': config.penalty_ship_no_messages,
        'penalty_explore': config.penalty_explore,
        
        # Idle penalties
        'penalty_idle_at_ship': config.penalty_idle_at_ship,
        'penalty_idle_at_sensor': config.penalty_idle_at_sensor,
        
        # Buffer penalties
        'penalty_buffer_overflow': config.penalty_buffer_overflow,
        'penalty_buffer_near_full': config.penalty_buffer_near_full,
        
        # TTL penalties
        'penalty_message_expired': config.penalty_message_expired,
        'penalty_message_expired_at_sensor': config.penalty_message_expired_at_sensor,
    }
    
    # DQN hyperparameters
    hyperparams = {
        'dqn_learning_rate': config.dqn_learning_rate,
        'dqn_buffer_size': config.dqn_buffer_size,
        'dqn_learning_starts': config.dqn_learning_starts,
        'dqn_batch_size': config.dqn_batch_size,
        'dqn_tau': config.dqn_tau,
        'dqn_gamma': config.dqn_gamma,
        'dqn_train_freq': config.dqn_train_freq,
        'dqn_gradient_steps': config.dqn_gradient_steps,
        'dqn_target_update_interval': config.dqn_target_update_interval,
        'dqn_exploration_fraction': config.dqn_exploration_fraction,
        'dqn_exploration_initial_eps': config.dqn_exploration_initial_eps,
        'dqn_exploration_final_eps': config.dqn_exploration_final_eps,
        'dqn_total_timesteps': config.dqn_total_timesteps,
        'dqn_eval_freq': config.dqn_eval_freq,
        'dqn_n_eval_episodes': config.dqn_n_eval_episodes,
        'dqn_net_arch': config.dqn_net_arch,  # ✅ ADDED
    }
    
    # Environment parameters
    env_params = {
        'area_size': config.area_size,
        'depth_range': config.depth_range,
        'num_sensors': config.num_sensors,
        'num_ships': config.num_ships,
        'drone_buffer_capacity': config.drone_buffer_capacity,
        'sensor_buffer_capacity': config.sensor_buffer_capacity,
        'message_ttl': config.message_ttl,
        'max_episode_steps': config.max_episode_steps,
        'sensors_state_space': config.sensors_state_space,
        'ships_state_space': config.ships_state_space,
    }
    
    return {
        'reward_parameters': rl_params,
        'dqn_hyperparameters': hyperparams,
        'environment_parameters': env_params
    }


def save_training_config(config: SimulationConfig, log_dir: str, model_name: str):
    """Save configuration when starting training"""
    
    config_data = {
        'model_name': model_name,
        'training_started': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'configuration': extract_rl_config(config)
    }
    
    # Save to model directory
    config_path = Path(log_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Also save to centralized configs folder
    centralized_path = Path("results/configs") / f"{model_name}.json"
    centralized_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(centralized_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n📝 Training configuration saved:")
    print(f"   Model directory: {config_path}")
    print(f"   Centralized: {centralized_path}")


def save_test_config(config: SimulationConfig, model_path: str, 
                     save_dir: str = "results/configs"):
    """
    Save configuration snapshot when testing a model.
    Saves to BOTH model directory AND centralized results folder.
    """
    
    # Get model name from path
    model_name = get_model_name_from_path(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config_data = {
        'timestamp': timestamp,
        'model_path': model_path,
        'model_name': model_name,
        'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'configuration': extract_rl_config(config)
    }
    
    # 1. Save to centralized results folder (for research/comparison)
    results_path = Path(save_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    results_filename = f"{model_name}.json"  # Use model name for easy lookup
    results_filepath = results_path / results_filename
    
    if results_filepath.exists():
        print(f"📝 Updating centralized config: {results_filepath}")
    else:
        print(f"📝 Creating centralized config: {results_filepath}")
    
    with open(results_filepath, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # 2. Save to model's directory (for self-contained storage)
    model_dir = Path(model_path).parent.parent
    model_config_path = model_dir / "test_config.json"
    
    with open(model_config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"📝 Config also saved to model: {model_config_path}")
    
    return results_filepath  # Return centralized path


def print_config_summary(config: SimulationConfig):
    """Print human-readable configuration summary"""
    
    rl_config = extract_rl_config(config)
    
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION SUMMARY")
    print("="*60)
    
    print("\n💰 REWARD PARAMETERS:")
    for key, value in rl_config['reward_parameters'].items():
        if 'reward' in key:
            print(f"   {key}: {value:,.1f}")
    
    print("\n⚠️  PENALTY PARAMETERS:")
    for key, value in rl_config['reward_parameters'].items():
        if 'penalty' in key:
            print(f"   {key}: {value:,.4f}" if abs(value) < 1 else f"   {key}: {value:,.1f}")
    
    print("\n🤖 DQN HYPERPARAMETERS:")
    for key, value in rl_config['dqn_hyperparameters'].items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}" if value < 0.01 else f"   {key}: {value:.4f}")
        elif isinstance(value, list):  # ✅ Handle list (for net_arch)
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:,}")
    
    print("\n🌍 ENVIRONMENT PARAMETERS:")
    for key, value in rl_config['environment_parameters'].items():
        if isinstance(value, (list, tuple)):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:,}" if isinstance(value, int) else f"   {key}: {value}")
    
    print("="*60 + "\n")


def save_test_results(trained_stats: dict, random_stats: dict, 
                      config_file: str, model_path: str,
                      save_dir: str = "results/test_results"):
    """
    Save test results to BOTH model directory AND centralized results folder.
    """
    
    model_name = get_model_name_from_path(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_data = {
        'timestamp': timestamp,
        'model_path': model_path,
        'model_name': model_name,
        'config_file': config_file,
        'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'trained_agent': {
            'mean_reward': float(trained_stats['mean_reward']),
            'messages_delivered': float(trained_stats['messages_delivered']),
            'delivery_rate': float(trained_stats['delivery_rate']),
            'mean_delivered_aoi': float(trained_stats['mean_delivered_aoi']),
            'mean_system_aoi': float(trained_stats['mean_system_aoi']),
            'undelivered_messages': float(trained_stats['undelivered_messages']),
            'sensor_aoi_avg': float(trained_stats['sensor_aoi_avg']),
            'drone_aoi_avg': float(trained_stats['drone_aoi_avg']),
        },
        'random_policy': {
            'mean_reward': float(random_stats['mean_reward']),
            'messages_delivered': float(random_stats['messages_delivered']),
            'delivery_rate': float(random_stats['delivery_rate']),
            'mean_delivered_aoi': float(random_stats['mean_delivered_aoi']),
            'mean_system_aoi': float(random_stats['mean_system_aoi']),
            'undelivered_messages': float(random_stats['undelivered_messages']),
            'sensor_aoi_avg': float(random_stats['sensor_aoi_avg']),
            'drone_aoi_avg': float(random_stats['drone_aoi_avg']),
        },
        'improvements': {
            'aoi_reduction_percent': float(
                (random_stats['mean_system_aoi'] - trained_stats['mean_system_aoi']) / 
                random_stats['mean_system_aoi'] * 100
            ) if random_stats['mean_system_aoi'] > 0 else 0.0,
            'delivery_rate_change_percent': float(
                (trained_stats['delivery_rate'] - random_stats['delivery_rate']) * 100
            ),
            'reward_improvement_percent': float(
                (trained_stats['mean_reward'] - random_stats['mean_reward']) / 
                abs(random_stats['mean_reward']) * 100
            ) if random_stats['mean_reward'] != 0 else 0.0
        }
    }
    
    # 1. Save to centralized results folder (for research/comparison)
    results_path = Path(save_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    results_filename = f"{model_name}.json"  # Use model name for easy lookup
    results_filepath = results_path / results_filename
    
    if results_filepath.exists():
        print(f"\n📊 Updating centralized results: {results_filepath}")
    else:
        print(f"\n📊 Creating centralized results: {results_filepath}")
    
    with open(results_filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # 2. Save to model's directory (for self-contained storage)
    model_dir = Path(model_path).parent.parent
    model_results_path = model_dir / "test_results.json"
    
    with open(model_results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"📊 Results also saved to model: {model_results_path}")
    print(f"📝 Linked to config: {config_file}")
    
    return results_filepath  # Return centralized path


def load_model_config(model_name: str, logs_dir: str = "./logs/single_agent") -> dict:
    """
    Load configuration for a specific model by name.
    
    Args:
        model_name: Name of the model (e.g., 'baseline_v1_20251121_143022')
        logs_dir: Directory where models are stored
        
    Returns:
        dict: Model configuration
    """
    model_dir = Path(logs_dir) / model_name
    config_path = model_dir / "config.json"
    
    if not config_path.exists():
        # Try test_config.json as fallback
        config_path = model_dir / "test_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for model '{model_name}' at {model_dir}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def list_all_models(logs_dir: str = "./logs/single_agent"):
    """List all available trained models"""
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"No models directory found at {logs_dir}")
        return []
    
    models = []
    for item in logs_path.iterdir():
        if item.is_dir():
            # Check if it has a model file
            has_model = (
                (item / "best_model" / "best_model.zip").exists() or
                (item / "final_model.zip").exists()
            )
            
            if has_model:
                # Try to load metadata
                metadata_path = item / "metadata.json"
                config_path = item / "config.json"
                results_path = item / "test_results.json"
                
                model_info = {
                    'name': item.name,
                    'path': str(item),
                    'has_metadata': metadata_path.exists(),
                    'has_config': config_path.exists(),
                    'has_results': results_path.exists()
                }
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_info['completed'] = metadata.get('training_completed', 'Unknown')
                
                models.append(model_info)
    
    return sorted(models, key=lambda x: x['name'], reverse=True)  # Most recent first