# compare_baselines.py

import simpy
import copy
import random
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

from src.simulation.agent_factory import AgentFactory
from src.simulation.processes import (
    sensor_process,
    ship_process,
    drone_process,
)
from src.movement.random_movement import RandomMovementStrategy
from src.movement.round_robin_movement import RoundRobinMovementStrategy
from src.movement.rl_movement import RLMovementStrategy
from analysis.metrics import MetricsCollector

BASE_SEED = 0


def run_simulation(config, movement_strategy, duration=86400, seed=None) -> Dict:
    cfg = copy.deepcopy(config)
    cfg.sim_time = duration

    env = simpy.Environment()
    factory = AgentFactory(cfg, drone_strategy=movement_strategy)
    sensors, drones, ships = factory.create_all_agents()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    assert len(drones) == 1, "This comparison script assumes 1 drone"
    assert len(ships) == 1, "This comparison script assumes 1 ship"

    drone = drones[0]
    ship = ships[0]

    # Metrics
    ship.metrics = MetricsCollector()
    ship.metrics.last_delivery_time_per_sensor = {s.id: 0.0 for s in sensors}
    ship.metrics.last_aoi_update_time = 0.0

    # Processes (pass metrics where needed)
    for sensor in sensors:
        env.process(sensor_process(env, sensor, cfg, ship.metrics))

    env.process(ship_process(env, ship, cfg))

    # IMPORTANT: pass metrics into drone_process so drone buffer drops count
    env.process(drone_process(env, drone, sensors, [ship], cfg, ship.metrics))

    env.run(until=duration)

    metrics = ship.get_episode_metrics(final_time=duration)

    # Useful derived metrics
    time_avg_aoi = metrics.aoi_integral / (duration * len(sensors))

    mean_delivered_message_age = (
        float(np.mean([m.aoi for m in ship.metrics.messages]))
        if ship.metrics.messages else float("inf")
    )

    # Calculate sensor coverage: fraction of sensors with at least one delivery
    sensors_with_delivery = set()
    for msg in ship.metrics.messages:
        sensors_with_delivery.add(msg.source_sensor)
    
    sensor_coverage = len(sensors_with_delivery) / len(sensors)

    return {
        "seed": seed,
        "aoi_integral": metrics.aoi_integral,
        "time_avg_aoi": time_avg_aoi,
        "mean_delivered_message_age": mean_delivered_message_age,
        "delivery_rate": metrics.delivery_rate,
        "delivered": metrics.delivered,
        "expired": metrics.expired,
        "generated": metrics.generated,
        "dropped_sensor_buffer": metrics.dropped_sensor_buffer,
        "dropped_drone_buffer": metrics.dropped_drone_buffer,
        "ship_visits": metrics.ship_visits,
        "sensor_visits_total": metrics.sensor_visits_total,
        "sensor_visits_per_sensor": metrics.sensor_visits_per_sensor,
        "sensor_coverage": sensor_coverage,
    }


def run_multiple_seeds(config, movement_strategy_factory, num_runs, duration, base_seed) -> List[Dict]:
    """
    Run multiple simulations with different seeds.
    
    Args:
        movement_strategy_factory: Either a strategy instance (for non-RL) or a callable that creates new instances (for RL)
    """
    results = []
    for i in range(num_runs):
        seed = base_seed + i
        
        # For RL strategies, create a fresh instance for each run to avoid state leakage
        # For others, reset and reuse
        if callable(movement_strategy_factory):
            strategy = movement_strategy_factory()
        else:
            strategy = movement_strategy_factory
            if hasattr(strategy, 'reset'):
                strategy.reset()
        
        results.append(run_simulation(config, strategy, duration, seed))
    return results


def summarize(results: List[Dict]) -> Dict:
    def mean_std(xs):
        xs = np.array(xs, dtype=float)
        return float(xs.mean()), float(xs.std(ddof=1))  # Sample std

    out = {}
    for key in [
        "aoi_integral",
        "time_avg_aoi",
        "mean_delivered_message_age",
        "delivery_rate",
        "delivered",
        "expired",
        "generated",
        "dropped_sensor_buffer",
        "dropped_drone_buffer",
        "ship_visits",
        "sensor_visits_total",
        "sensor_coverage",
    ]:
        m, s = mean_std([r[key] for r in results])
        out[key] = {"mean": m, "std": s}
    
    # Per-sensor visit stats (aggregate across runs)
    if "sensor_visits_per_sensor" in results[0]:
        all_sensor_ids = set()
        for r in results:
            all_sensor_ids.update(r["sensor_visits_per_sensor"].keys())
        
        per_sensor_stats = {}
        for sid in sorted(all_sensor_ids):
            visits = [r["sensor_visits_per_sensor"].get(sid, 0) for r in results]
            m, s = mean_std(visits)
            per_sensor_stats[sid] = {"mean": m, "std": s}
        
        out["sensor_visits_per_sensor"] = per_sensor_stats
    
    return out


def print_summary(name: str, stats: Dict):
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in stats.items():
        if k == "sensor_visits_per_sensor":
            # Skip per-sensor details in summary (or print separately)
            continue
        print(f"{k:24s} {v['mean']:.6g}  ± {v['std']:.6g}")
    
    # Optionally print per-sensor visit distribution
    if "sensor_visits_per_sensor" in stats:
        print("\n  Per-sensor visits:")
        for sensor_id, sensor_stats in sorted(stats["sensor_visits_per_sensor"].items()):
            print(f"    Sensor {sensor_id:2d}:  {sensor_stats['mean']:6.2f} ± {sensor_stats['std']:5.2f}")


def plot_comparison(all_results: Dict, strategies: Dict, output_dir: str, timestamp: str):
    """
    Create bar plots comparing key metrics across strategies.
    """
    # Set up matplotlib with academic paper style
    plt.rcParams.update({
        # ---- Font (Times family) ----
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes"],

        # ---- Font sizes ----
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # ---- Axis style ----
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # ---- Save settings ----
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    
    # Extract strategy names and data
    strategy_keys = list(strategies.keys())
    strategy_names = [strategies[key][1] for key in strategy_keys]
    
    # Metrics to plot (now 5 metrics in 2x3 grid, with one empty subplot)
    metrics = [
        ("aoi_integral", "AoI Integral", "Total AoI (s²)"),
        ("time_avg_aoi", "Time-Averaged AoI", "Average AoI (s)"),
        ("mean_delivered_message_age", "Mean Delivered Message Age", "Age (s)"),
        ("delivery_rate", "Delivery Rate", "Rate"),
        ("sensor_coverage", "Sensor Coverage", "Fraction")
    ]
    
    # First pass: collect all data to determine consistent y-limits
    metric_data = {}
    for metric_key, title, ylabel in metrics:
        means = []
        stds = []
        for key in strategy_keys:
            stats = all_results[key]["stats"]
            means.append(stats[metric_key]["mean"])
            stds.append(stats[metric_key]["std"])
        
        # Calculate y-limit: max value + error bar + 10% padding
        max_val = max([m + s for m, s in zip(means, stds)])
        y_limit = max_val * 1.15  # 15% padding for labels
        
        metric_data[metric_key] = {
            'means': means,
            'stds': stds,
            'y_limit': y_limit
        }
    
    # Create figure with subplots (2x3 grid, one will be empty)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 4.8))  # 3 columns x 3.4 width
    axes = axes.flatten()
    
    # Use distinct colors for better visibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (metric_key, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        means = metric_data[metric_key]['means']
        stds = metric_data[metric_key]['stds']
        y_limit = metric_data[metric_key]['y_limit']
        
        x = np.arange(len(strategy_names))
        bars = ax.bar(x, means, yerr=stds, capsize=3, 
                     color=colors[:len(strategy_names)], 
                     alpha=0.8, edgecolor='black', linewidth=0.8)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        
        # Set consistent y-axis limits
        ax.set_ylim(0, y_limit)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            if metric_key == "aoi_integral":
                label = f'{mean:.2e}'
            elif metric_key in ["delivery_rate", "sensor_coverage"]:
                label = f'{mean:.3f}'
            else:
                label = f'{mean:.1f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=7)
    
    # Hide the last (6th) subplot since we only have 5 metrics
    axes[5].axis('off')
    
    fig.tight_layout()
    
    # Save as PDF
    pdf_path = os.path.join(output_dir, f"comparison_{timestamp}.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    print(f"\nPlot saved: {pdf_path}")
    
    # Display the plot
    plt.show()


def discover_rl_models(models_dir="./models"):
    """
    Discover all trained RL models in the models directory.
    Returns list of (model_path, model_name) tuples.
    """
    models = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return models
    
    # Look for any subdirectory with a best_model.zip
    for model_dir in sorted(models_path.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
            
        best_model = model_dir / "best_model.zip"
        if best_model.exists():
            models.append((str(best_model), model_dir.name))
        else:
            # Also check for final model if best doesn't exist
            final_model = model_dir / "ppo_drone_final.zip"
            if final_model.exists():
                models.append((str(final_model), model_dir.name))
    
    return models


if __name__ == "__main__":
    from src.config.simulation_config import SimulationConfig

    parser = argparse.ArgumentParser(description="Compare baseline movement strategies")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per strategy")
    parser.add_argument("--duration", type=int, default=86400, help="Simulation duration in seconds")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED, help="Base seed")
    parser.add_argument("--outdir", type=str, default="results/baseline_comparison", help="Output directory")
    parser.add_argument("--rl-model", type=str, default=None, 
                        help="Path to specific RL model (.zip file)")
    parser.add_argument("--rl-episode-duration", type=int, default=86400,
                        help="Episode duration used during RL training (default: 86400s = 24h)")
    parser.add_argument("--all-rl-models", action="store_true",
                        help="Include all discovered RL models in models/ directory")
    parser.add_argument("--models-dir", type=str, default="./models",
                        help="Directory containing RL models (default: ./models)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    
    args = parser.parse_args()

    config = SimulationConfig()

    # Build strategies dict
    strategies = {}
    
    random_strategy = RandomMovementStrategy()
    rr_strategy = RoundRobinMovementStrategy()
    
    strategies["random"] = (random_strategy, "Random Movement")
    strategies["round_robin"] = (rr_strategy, "Round Robin")
    
    # Add RL models
    rl_models_added = []
    
    if args.all_rl_models:
        # Discover all models
        discovered_models = discover_rl_models(args.models_dir)
        
        if discovered_models:
            print(f"\nDiscovered {len(discovered_models)} RL model(s):")
            for model_path, model_name in discovered_models:
                print(f"  - {model_name}: {model_path}")
            print()
            
            for model_path, model_name in discovered_models:
                try:
                    # Create a factory function that creates fresh RLMovementStrategy instances
                    # This prevents state leakage between runs
                    def make_rl_strategy(path=model_path, duration=args.rl_episode_duration):
                        return RLMovementStrategy(
                            model_path=path,
                            episode_duration=duration
                        )
                    
                    # Test that it loads successfully
                    test_strategy = make_rl_strategy()
                    print(f"✓ Loaded {model_name}: {type(test_strategy.model).__name__}")
                    
                    # Use model_name as key (e.g., "run_20251214_143245")
                    strategies[f"rl_{model_name}"] = (make_rl_strategy, f"RL: {model_name}")
                    rl_models_added.append(model_name)
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
        else:
            print(f"\nNo RL models found in {args.models_dir}/")
    
    elif args.rl_model:
        # Load specific model
        try:
            # Create a factory function
            def make_rl_strategy(path=args.rl_model, duration=args.rl_episode_duration):
                return RLMovementStrategy(
                    model_path=path,
                    episode_duration=duration
                )
            
            # Test that it loads successfully
            test_strategy = make_rl_strategy()
            
            # Extract name from path
            model_name = Path(args.rl_model).stem  # e.g., "best_model"
            parent_name = Path(args.rl_model).parent.name  # e.g., "run_20251214_143245"
            full_name = f"{parent_name}_{model_name}" if parent_name != "." else model_name
            
            strategies["rl"] = (make_rl_strategy, f"RL: {full_name}")
            print(f"Loaded RL model: {args.rl_model}")
            print(f"   Model type: {type(test_strategy.model).__name__}")
            print(f"   Episode duration: {args.rl_episode_duration}s ({args.rl_episode_duration/3600:.1f}h)\n")
            rl_models_added.append(full_name)
        except Exception as e:
            print(f"Failed to load RL model: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDuration: {args.duration}s | Runs: {args.num_runs} | Seeds: {args.base_seed}..{args.base_seed + args.num_runs - 1}")
    print(f"Sensors: {config.num_sensors} | Drones: {config.num_drones}")
    print(f"Strategies: {', '.join([name for _, name in strategies.values()])}\n")

    # RUN ALL STRATEGIES
    all_results = {}
    for key, (strategy, name) in strategies.items():
        print(f"\n{'='*60}")
        print(f"Running {name}...")
        print(f"{'='*60}")
        runs = run_multiple_seeds(config, strategy, args.num_runs, args.duration, args.base_seed)
        all_results[key] = {"stats": summarize(runs), "runs": runs}

    # PRINT ALL SUMMARIES
    for key, (_, name) in strategies.items():
        print_summary(name, all_results[key]["stats"])

    # SAVE JSON
    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": timestamp,
        "duration": args.duration,
        "num_runs": args.num_runs,
        "base_seed": args.base_seed,
        "rl_models_included": rl_models_added,
        **all_results  # Unpack all strategy results
    }
    path = os.path.join(args.outdir, f"comparison_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")
    
    # GENERATE PLOTS
    if not args.no_plot:
        plot_comparison(all_results, strategies, args.outdir, timestamp)