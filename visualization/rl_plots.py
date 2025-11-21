"""Visualization tools for RL training and testing results"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def get_model_visualization_dir(model_path: str) -> Path:
    """
    Get visualization directory for a specific model.
    This is a standalone function (not a class method).
    
    Args:
        model_path: Path to model file (e.g., logs/single_agent/baseline_v1/best_model/best_model.zip)
        
    Returns:
        Path to model's visualization directory
    """
    # Model path: logs/single_agent/baseline_v1_20251121/best_model/best_model.zip
    # Want: logs/single_agent/baseline_v1_20251121/visualizations/
    model_dir = Path(model_path).parent.parent
    vis_dir = model_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    return vis_dir


class RLResultsVisualizer:
    """Visualize RL training and testing results"""
    
    def __init__(self, save_dir: str = "results/plots/rl"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_comparison_metrics(self, trained_stats: Dict, random_stats: Dict):
        """Create comprehensive comparison charts between trained and random policies"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🏆 Trained Agent vs Random Policy Comparison', fontsize=16, fontweight='bold')
        
        # 1. Total Reward Comparison
        ax = axes[0, 0]
        rewards = [trained_stats['mean_reward'], random_stats['mean_reward']]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['Trained Agent', 'Random Policy'], rewards, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Episode Reward', fontweight='bold')
        ax.set_title('💰 Total Reward Comparison')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:,.0f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        improvement = ((trained_stats['mean_reward'] - random_stats['mean_reward']) / 
                      abs(random_stats['mean_reward']) * 100)
        ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 2. Delivery Rate Comparison
        ax = axes[0, 1]
        delivery_rates = [
            trained_stats['delivery_rate'] * 100,
            random_stats['delivery_rate'] * 100
        ]
        bars = ax.bar(['Trained Agent', 'Random Policy'], delivery_rates, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Delivery Rate (%)', fontweight='bold')
        ax.set_title('📨 Message Delivery Rate')
        ax.set_ylim([0, 100])
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        # 3. Messages Delivered
        ax = axes[0, 2]
        messages = [trained_stats['messages_delivered'], random_stats['messages_delivered']]
        bars = ax.bar(['Trained Agent', 'Random Policy'], messages, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Messages Delivered', fontweight='bold')
        ax.set_title('📬 Total Messages Delivered')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontweight='bold')
        
        # 4. System-Wide AoI (KEY METRIC!)
        ax = axes[1, 0]
        system_aoi = [
            trained_stats['mean_system_aoi'],
            random_stats['mean_system_aoi']
        ]
        bars = ax.bar(['Trained Agent', 'Random Policy'], system_aoi, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean System AoI (seconds)', fontweight='bold')
        ax.set_title('🌐 System-Wide AoI (Lower is Better) ⭐')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}s',
                   ha='center', va='bottom', fontweight='bold')
        
        aoi_reduction = ((random_stats['mean_system_aoi'] - trained_stats['mean_system_aoi']) / 
                        random_stats['mean_system_aoi'] * 100)
        ax.text(0.5, 0.95, f'AoI Reduction: {aoi_reduction:.1f}% ✅', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
               fontweight='bold', fontsize=12)
        
        # 5. Delivered Message AoI
        ax = axes[1, 1]
        delivered_aoi = [
            trained_stats['mean_delivered_aoi'],
            random_stats['mean_delivered_aoi']
        ]
        bars = ax.bar(['Trained Agent', 'Random Policy'], delivered_aoi, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Delivered AoI (seconds)', fontweight='bold')
        ax.set_title('⏱️ Delivered Messages AoI')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}s',
                   ha='center', va='bottom', fontweight='bold')
        
        if random_stats['mean_delivered_aoi'] > 0:
            delivered_aoi_reduction = ((random_stats['mean_delivered_aoi'] - trained_stats['mean_delivered_aoi']) / 
                                      random_stats['mean_delivered_aoi'] * 100)
            ax.text(0.5, 0.95, f'Reduction: {delivered_aoi_reduction:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 6. Undelivered Messages
        ax = axes[1, 2]
        undelivered = [
            trained_stats['undelivered_messages'],
            random_stats['undelivered_messages']
        ]
        bars = ax.bar(['Trained Agent', 'Random Policy'], undelivered, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Undelivered Messages', fontweight='bold')
        ax.set_title('📭 Messages Left Undelivered')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save to self.save_dir (will be overridden by create_test_visualization)
        filepath = self.save_dir / 'test_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison to {filepath}")
        plt.close()
    
    def plot_aoi_breakdown(self, trained_stats: Dict, random_stats: Dict):
        """Plot detailed AoI breakdown"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('📊 Detailed AoI Analysis', fontsize=14, fontweight='bold')
        
        # Trained Agent AoI Breakdown
        ax = axes[0]
        categories = ['Delivered\nMessages', 'Sensor\nMessages', 'Drone\nBuffer']
        trained_aoi = [
            trained_stats.get('mean_delivered_aoi', 0),
            trained_stats.get('sensor_aoi_avg', 0),
            trained_stats.get('drone_aoi_avg', 0)
        ]
        
        bars = ax.bar(categories, trained_aoi, color=['#3498db', '#e67e22', '#9b59b6'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Average AoI (seconds)', fontweight='bold')
        ax.set_title('Trained Agent - AoI Distribution')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}s',
                       ha='center', va='bottom', fontweight='bold')
        
        # Random Policy AoI Breakdown
        ax = axes[1]
        random_aoi = [
            random_stats.get('mean_delivered_aoi', 0),
            random_stats.get('sensor_aoi_avg', 0),
            random_stats.get('drone_aoi_avg', 0)
        ]
        
        bars = ax.bar(categories, random_aoi, color=['#3498db', '#e67e22', '#9b59b6'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Average AoI (seconds)', fontweight='bold')
        ax.set_title('Random Policy - AoI Distribution')
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}s',
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save to self.save_dir
        filepath = self.save_dir / 'test_metrics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved AoI breakdown to {filepath}")
        plt.close()


def create_test_visualization(trained_stats: Dict, random_stats: Dict, model_path: str = None):
    """
    Create all visualizations from test results.
    
    Args:
        trained_stats: Statistics from trained agent
        random_stats: Statistics from random policy
        model_path: Path to model file. If provided, saves to model's visualization directory.
                   If None, saves to default results/plots/rl directory.
    """
    
    # Determine save directory
    if model_path:
        save_dir = get_model_visualization_dir(model_path)
        print(f"\n📊 Saving visualizations to model directory: {save_dir}")
    else:
        save_dir = Path("results/plots/rl")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📊 Saving visualizations to default directory: {save_dir}")
    
    # Create visualizer with appropriate save directory
    visualizer = RLResultsVisualizer(save_dir=str(save_dir))
    
    # Generate all plots
    visualizer.plot_comparison_metrics(trained_stats, random_stats)
    visualizer.plot_aoi_breakdown(trained_stats, random_stats)
    
    print("\n" + "="*60)
    print("✅ All visualizations created successfully!")
    print(f"📁 Saved to: {save_dir}")
    print("="*60)