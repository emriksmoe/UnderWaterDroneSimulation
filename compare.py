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
    }


def run_multiple_seeds(config, movement_strategy, num_runs, duration, base_seed) -> List[Dict]:
    results = []
    for i in range(num_runs):
        seed = base_seed + i
        
        # Reset strategy state before each run
        if hasattr(movement_strategy, 'reset'):
            movement_strategy.reset()
        
        results.append(run_simulation(config, movement_strategy, duration, seed))
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
                    rl_strategy = RLMovementStrategy(
                        model_path=model_path,
                        episode_duration=args.rl_episode_duration
                    )
                    # Use model_name as key (e.g., "run_20251214_143245")
                    strategies[f"rl_{model_name}"] = (rl_strategy, f"RL: {model_name}")
                    rl_models_added.append(model_name)
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
        else:
            print(f"\nNo RL models found in {args.models_dir}/")
    
    elif args.rl_model:
        # Load specific model
        try:
            rl_strategy = RLMovementStrategy(
                model_path=args.rl_model,
                episode_duration=args.rl_episode_duration
            )
            # Extract name from path
            model_name = Path(args.rl_model).stem  # e.g., "best_model"
            parent_name = Path(args.rl_model).parent.name  # e.g., "run_20251214_143245"
            full_name = f"{parent_name}_{model_name}" if parent_name != "." else model_name
            
            strategies["rl"] = (rl_strategy, f"RL: {full_name}")
            print(f"Loaded RL model: {args.rl_model}")
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