"""Compare results across different configurations"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_test_result(filepath: str) -> dict:
    """Load a single test result file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_results(results_dir: str = "results/test_results") -> List[dict]:
    """Load all test results from centralized directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"⚠️ No centralized results found at {results_dir}")
        print("   Results are stored in individual model directories")
        return []
    
    results = []
    for filepath in results_path.glob("*.json"):
        try:
            result = load_test_result(filepath)
            results.append(result)
        except Exception as e:
            print(f"⚠️ Failed to load {filepath}: {e}")
    
    return results


def load_configs_for_models(model_names: List[str]) -> List[Tuple[str, dict]]:
    """Load configurations for a list of model names"""
    from src.utils.config_logger import load_model_config
    
    configs = []
    for name in model_names:
        try:
            config = load_model_config(name)
            configs.append((name, config))
        except FileNotFoundError:
            print(f"⚠️ Warning: Config not found for {name}")
    
    return configs


def compare_results(results_dir: str = "results/test_results", show_plot: bool = True, show_config_diff: bool = True):
    """Compare all test results and show best configurations"""
    
    results = load_all_results(results_dir)
    
    if not results:
        print("No test results found in centralized storage!")
        print("Test some models first with: python -m src.rl.testing.test_single_agent_model")
        return None
    
    # Create comparison dataframe
    comparison_data = []
    
    for result in results:
        row = {
            'model_name': result['model_name'],
            'test_date': result.get('test_date', 'Unknown'),
            'aoi_reduction_%': result['improvements']['aoi_reduction_percent'],
            'delivery_rate_%': result['trained_agent']['delivery_rate'] * 100,
            'mean_reward': result['trained_agent']['mean_reward'],
            'system_aoi': result['trained_agent']['mean_system_aoi'],
            'messages_delivered': result['trained_agent']['messages_delivered'],
            'undelivered': result['trained_agent']['undelivered_messages'],
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('aoi_reduction_%', ascending=False)
    
    print("\n" + "="*100)
    print("📊 TEST RESULTS COMPARISON (Sorted by AoI Reduction)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # Show best configuration
    if len(df) > 0:
        best_idx = df['aoi_reduction_%'].idxmax()
        best = df.iloc[best_idx]
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Model: {best['model_name']}")
        print(f"   Test Date: {best['test_date']}")
        print(f"   AoI Reduction: {best['aoi_reduction_%']:.1f}%")
        print(f"   Delivery Rate: {best['delivery_rate_%']:.1f}%")
        print(f"   Mean Reward: {best['mean_reward']:,.0f}")
        print(f"   System AoI: {best['system_aoi']:.1f}s")
        
        # Load and display best config
        config_path = Path("results/configs") / f"{best['model_name']}.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                best_config = json.load(f)
            
            print(f"\n📝 Best Configuration Key Parameters:")
            rewards = best_config['configuration']['reward_parameters']
            print(f"   reward_delivery_base: {rewards['reward_delivery_base']:,.0f}")
            print(f"   penalty_undelivered_base: {rewards['penalty_undelivered_base']:,.0f}")
            print(f"   penalty_carrying_per_age_unit: {rewards['penalty_carrying_per_age_unit']:.4f}")
    
    # Show configuration differences if requested and we have 2+ models
    if show_config_diff and len(df) >= 2:
        print("\n" + "="*100)
        print("🔍 CONFIGURATION DIFFERENCES ACROSS ALL MODELS")
        print("="*100)
        
        model_names = df['model_name'].tolist()
        configs = load_configs_for_models(model_names)
        
        if len(configs) >= 2:
            differences = find_config_differences(configs)
            
            if differences:
                _show_differences_summary(differences, configs)
            else:
                print("\n✅ All model configurations are IDENTICAL!")
        else:
            print("\n⚠️ Could not load configs for comparison")
    
    # Create visualization if requested
    if show_plot and len(df) > 0:
        create_comparison_plots(df, results_dir)
    
    return df


def _show_differences_summary(differences: Dict[str, Dict[str, any]], configs: List[Tuple[str, dict]]):
    """Show a summary of configuration differences"""
    
    # Group differences by category
    reward_diffs = {k: v for k, v in differences.items() if 'reward_parameters' in k}
    hyperparams_diffs = {k: v for k, v in differences.items() if 'dqn_hyperparameters' in k}
    env_diffs = {k: v for k, v in differences.items() if 'environment_parameters' in k}
    
    print(f"\n🔴 Found {len(differences)} DIFFERENT parameters:")
    print(f"   💰 {len(reward_diffs)} reward parameters")
    print(f"   🤖 {len(hyperparams_diffs)} DQN hyperparameters")
    print(f"   🌍 {len(env_diffs)} environment parameters")
    
    # Show top differences in rewards
    if reward_diffs:
        print("\n" + "-"*100)
        print("💰 KEY REWARD PARAMETER DIFFERENCES:")
        print("-"*100)
        
        # Show first 10 or all if less
        show_count = min(10, len(reward_diffs))
        for i, (param_path, values) in enumerate(sorted(reward_diffs.items())[:show_count]):
            param_name = param_path.split('.')[-1]
            print(f"\n{i+1}. {param_name}:")
            
            for model_name, value in values.items():
                if isinstance(value, float):
                    formatted = f"{value:,.4f}" if abs(value) < 1 else f"{value:,.1f}"
                else:
                    formatted = str(value)
                
                # Highlight if this is min/max
                all_values = [v for v in values.values() if isinstance(v, (int, float))]
                if all_values and len(all_values) > 1:
                    if value == min(all_values):
                        print(f"   🔽 {model_name[:30]:30s}: {formatted:>15s} (MIN)")
                    elif value == max(all_values):
                        print(f"   🔼 {model_name[:30]:30s}: {formatted:>15s} (MAX)")
                    else:
                        print(f"      {model_name[:30]:30s}: {formatted:>15s}")
                else:
                    print(f"      {model_name[:30]:30s}: {formatted:>15s}")
        
        if len(reward_diffs) > show_count:
            print(f"\n   ... and {len(reward_diffs) - show_count} more differences")
            print(f"   Use: python -m src.utils.compare_config compare MODEL1 MODEL2 ... for full details")
    
    # Show hyperparameter differences if any
    if hyperparams_diffs:
        print("\n" + "-"*100)
        print("🤖 DQN HYPERPARAMETER DIFFERENCES:")
        print("-"*100)
        
        for param_path, values in sorted(hyperparams_diffs.items()):
            param_name = param_path.split('.')[-1]
            print(f"\n{param_name}:")
            
            for model_name, value in values.items():
                if isinstance(value, float):
                    formatted = f"{value:.6f}"
                else:
                    formatted = f"{value:,}"
                print(f"   {model_name[:30]:30s}: {formatted:>15s}")
    
    # Show environment differences if any
    if env_diffs:
        print("\n" + "-"*100)
        print("🌍 ENVIRONMENT PARAMETER DIFFERENCES:")
        print("-"*100)
        
        for param_path, values in sorted(env_diffs.items()):
            param_name = param_path.split('.')[-1]
            print(f"\n{param_name}:")
            
            for model_name, value in values.items():
                print(f"   {model_name[:30]:30s}: {str(value):>15s}")
    
    print("\n" + "="*100)
    print(f"💡 Tip: For detailed side-by-side comparison, use:")
    print(f"   python -m src.utils.compare_config compare MODEL1 MODEL2")
    print("="*100)


def create_comparison_plots(df: pd.DataFrame, results_dir: str):
    """Create visualization comparing all models"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Sort by model name for consistent ordering
    df_sorted = df.sort_values('model_name')
    
    # 1. AoI Reduction comparison
    ax1 = axes[0, 0]
    bars1 = ax1.barh(df_sorted['model_name'], df_sorted['aoi_reduction_%'], 
                     color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('AoI Reduction (%)', fontweight='bold')
    ax1.set_title('AoI Reduction by Model', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Highlight best
    best_idx = df_sorted['aoi_reduction_%'].idxmax()
    bars1[list(df_sorted.index).index(best_idx)].set_color('#27ae60')
    bars1[list(df_sorted.index).index(best_idx)].set_linewidth(2)
    
    # 2. Delivery Rate comparison
    ax2 = axes[0, 1]
    bars2 = ax2.barh(df_sorted['model_name'], df_sorted['delivery_rate_%'],
                     color='#3498db', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Delivery Rate (%)', fontweight='bold')
    ax2.set_title('Delivery Rate by Model', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. System AoI comparison
    ax3 = axes[1, 0]
    bars3 = ax3.barh(df_sorted['model_name'], df_sorted['system_aoi'],
                     color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('System AoI (seconds)', fontweight='bold')
    ax3.set_title('System AoI by Model (Lower is Better)', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Highlight best (lowest)
    best_idx = df_sorted['system_aoi'].idxmin()
    bars3[list(df_sorted.index).index(best_idx)].set_color('#c0392b')
    bars3[list(df_sorted.index).index(best_idx)].set_linewidth(2)
    
    # 4. Scatter: AoI Reduction vs Delivery Rate
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_sorted['delivery_rate_%'], df_sorted['aoi_reduction_%'],
                         s=200, c=df_sorted['system_aoi'], cmap='RdYlGn_r',
                         alpha=0.7, edgecolor='black', linewidth=2)
    
    # Annotate points
    for idx, row in df_sorted.iterrows():
        ax4.annotate(row['model_name'], 
                    (row['delivery_rate_%'], row['aoi_reduction_%']),
                    fontsize=8, ha='right', va='bottom')
    
    ax4.set_xlabel('Delivery Rate (%)', fontweight='bold')
    ax4.set_ylabel('AoI Reduction (%)', fontweight='bold')
    ax4.set_title('AoI Reduction vs Delivery Rate', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('System AoI (s)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(results_dir).parent / "plots" / "model_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved: {plot_path}")
    
    plt.close()


def find_config_differences(configs: List[Tuple[str, dict]]) -> Dict[str, Dict[str, any]]:
    """
    Find parameters that differ between configurations.
    
    Args:
        configs: List of (model_name, config_dict) tuples
        
    Returns:
        Dictionary of {parameter_path: {model_name: value}}
    """
    differences = {}
    
    if len(configs) < 2:
        return differences
    
    # Get all parameter paths
    def get_all_paths(d, prefix=''):
        """Recursively get all parameter paths"""
        paths = {}
        for key, value in d.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths.update(get_all_paths(value, current_path))
            else:
                paths[current_path] = value
        return paths
    
    # Get paths for all configs
    all_params = {}
    for model_name, config in configs:
        params = get_all_paths(config['configuration'])
        all_params[model_name] = params
    
    # Find differences
    first_model = configs[0][0]
    first_params = all_params[first_model]
    
    for param_path, first_value in first_params.items():
        values_dict = {first_model: first_value}
        has_difference = False
        
        for model_name, config in configs[1:]:
            if param_path in all_params[model_name]:
                current_value = all_params[model_name][param_path]
                values_dict[model_name] = current_value
                
                # ✅ IMPROVED: Better comparison logic
                if _values_are_different(first_value, current_value):
                    has_difference = True
        
        if has_difference:
            differences[param_path] = values_dict
    
    return differences


def _values_are_different(val1, val2) -> bool:
    """
    Compare two values for differences.
    Handles floats, lists, and other types properly.
    """
    # Handle None cases
    if val1 is None or val2 is None:
        return val1 != val2
    
    # Handle lists
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return True
        return any(_values_are_different(v1, v2) for v1, v2 in zip(val1, val2))
    
    # Handle floats with tolerance
    if isinstance(val1, float) and isinstance(val2, float):
        return abs(val1 - val2) > 1e-9
    
    # Handle all other types
    return val1 != val2

def compare_configs_detailed(*model_names, show_all: bool = False):
    """
    Compare detailed configurations of specific models.
    
    Args:
        *model_names: Model names to compare
        show_all: If True, show all parameters. If False, show only differences.
    """
    
    if not model_names:
        print("Please specify model names to compare")
        print("\nUsage:")
        print("  python -m src.utils.compare_config compare MODEL1 MODEL2 [MODEL3 ...]")
        print("  python -m src.utils.compare_config compare MODEL1 MODEL2 --all  # Show all params")
        return
    
    configs = []
    
    for name in model_names:
        try:
            from src.utils.config_logger import load_model_config
            config = load_model_config(name)
            configs.append((name, config))
        except FileNotFoundError:
            print(f"⚠️ Skipping {name} - not found")
    
    if len(configs) < 2:
        print("Need at least 2 valid models to compare!")
        return
    
    print("\n" + "="*120)
    print(f"🔍 CONFIGURATION COMPARISON: {len(configs)} Models")
    print("="*120)
    
    if show_all:
        print("📋 Showing ALL parameters")
        _show_all_params(configs)
    else:
        print("⚠️  Showing ONLY DIFFERENT parameters (use --all to see everything)")
        _show_differences_only(configs)


def _show_differences_only(configs: List[Tuple[str, dict]]):
    """Show only parameters that differ between models"""
    
    differences = find_config_differences(configs)
    
    if not differences:
        print("\n✅ All configurations are IDENTICAL!")
        return
    
    # Group differences by category
    reward_diffs = {k: v for k, v in differences.items() if 'reward_parameters' in k}
    hyperparams_diffs = {k: v for k, v in differences.items() if 'dqn_hyperparameters' in k}
    env_diffs = {k: v for k, v in differences.items() if 'environment_parameters' in k}
    
    print(f"\n🔴 Found {len(differences)} DIFFERENT parameters")
    
    # Show reward parameter differences
    if reward_diffs:
        print("\n" + "="*120)
        print("💰 REWARD PARAMETERS (DIFFERENCES ONLY):")
        print("="*120)
        
        for param_path, values in sorted(reward_diffs.items()):
            param_name = param_path.split('.')[-1]  # Get last part of path
            print(f"\n📌 {param_name}:")
            
            for model_name, value in values.items():
                if isinstance(value, float):
                    formatted = f"{value:,.4f}" if abs(value) < 1 else f"{value:,.1f}"
                else:
                    formatted = str(value)
                
                # Highlight if this is min/max
                all_values = [v for v in values.values() if isinstance(v, (int, float))]
                if all_values:
                    if value == min(all_values):
                        print(f"   🔽 {model_name}: {formatted} (MIN)")
                    elif value == max(all_values):
                        print(f"   🔼 {model_name}: {formatted} (MAX)")
                    else:
                        print(f"      {model_name}: {formatted}")
                else:
                    print(f"      {model_name}: {formatted}")
    
    # Show hyperparameter differences
    if hyperparams_diffs:
        print("\n" + "="*120)
        print("🤖 DQN HYPERPARAMETERS (DIFFERENCES ONLY):")
        print("="*120)
        
        for param_path, values in sorted(hyperparams_diffs.items()):
            param_name = param_path.split('.')[-1]
            print(f"\n📌 {param_name}:")
            
            for model_name, value in values.items():
                if isinstance(value, float):
                    formatted = f"{value:.6f}"
                else:
                    formatted = f"{value:,}"
                
                all_values = [v for v in values.values() if isinstance(v, (int, float))]
                if all_values:
                    if value == min(all_values):
                        print(f"   🔽 {model_name}: {formatted} (MIN)")
                    elif value == max(all_values):
                        print(f"   🔼 {model_name}: {formatted} (MAX)")
                    else:
                        print(f"      {model_name}: {formatted}")
                else:
                    print(f"      {model_name}: {formatted}")
    
    # Show environment differences
    if env_diffs:
        print("\n" + "="*120)
        print("🌍 ENVIRONMENT PARAMETERS (DIFFERENCES ONLY):")
        print("="*120)
        
        for param_path, values in sorted(env_diffs.items()):
            param_name = param_path.split('.')[-1]
            print(f"\n📌 {param_name}:")
            
            for model_name, value in values.items():
                print(f"      {model_name}: {value}")
    
    print("\n" + "="*120)
    print(f"📊 Summary: {len(reward_diffs)} reward params, {len(hyperparams_diffs)} hyperparams, {len(env_diffs)} env params differ")
    print("="*120)


def _show_all_params(configs: List[Tuple[str, dict]]):
    """Show all parameters (original behavior)"""
    
    print("\n💰 REWARD PARAMETERS:")
    reward_keys = [
        'reward_collection_base',
        'reward_delivery_base', 
        'reward_delivery_freshness_multiplier',
        'penalty_undelivered_base', 
        'penalty_undelivered_age_multiplier',
        'penalty_carrying_per_age_unit',
        'penalty_buffer_near_full',
        'penalty_time_per_second',
        'penalty_empty_sensor',
        'penalty_ship_no_messages',
    ]
    
    for key in reward_keys:
        print(f"\n{key}:")
        for name, config in configs:
            value = config['configuration']['reward_parameters'].get(key, 'N/A')
            if isinstance(value, float):
                print(f"   {name}: {value:,.4f}" if abs(value) < 1 else f"   {name}: {value:,.1f}")
            else:
                print(f"   {name}: {value}")
    
    print("\n\n🤖 DQN HYPERPARAMETERS:")
    hyper_keys = ['dqn_learning_rate', 'dqn_gamma', 'dqn_buffer_size', 'dqn_total_timesteps']
    
    for key in hyper_keys:
        print(f"\n{key}:")
        for name, config in configs:
            value = config['configuration']['dqn_hyperparameters'].get(key, 'N/A')
            if isinstance(value, float):
                print(f"   {name}: {value:.6f}")
            else:
                print(f"   {name}: {value:,}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Detailed comparison mode
        model_names = [arg for arg in sys.argv[2:] if not arg.startswith('--')]
        show_all = '--all' in sys.argv
        
        compare_configs_detailed(*model_names, show_all=show_all)
    else:
        # Summary comparison mode
        compare_results()