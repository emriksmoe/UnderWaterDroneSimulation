#!/usr/bin/env python3
"""Debug script to see what decisions each RL model is making"""

import simpy
import copy
import numpy as np
from src.config.simulation_config import SimulationConfig
from src.simulation.agent_factory import AgentFactory
from src.simulation.processes import sensor_process, ship_process, drone_process
from src.movement.rl_movement import RLMovementStrategy
from analysis.metrics import MetricsCollector

def test_model_decisions(model_path, model_name, duration=3600, seed=42):
    """Run a short simulation and log the first 20 actions"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")
    
    config = SimulationConfig()
    config.sim_time = duration
    
    # Set seed
    np.random.seed(seed)
    
    # Create environment
    env = simpy.Environment()
    
    # Create RL strategy
    strategy = RLMovementStrategy(model_path=model_path, episode_duration=86400)
    
    # Create agents
    factory = AgentFactory(config, drone_strategy=strategy)
    sensors, drones, ships = factory.create_all_agents()
    
    drone = drones[0]
    ship = ships[0]
    
    # Metrics
    ship.metrics = MetricsCollector()
    ship.metrics.last_delivery_time_per_sensor = {s.id: 0.0 for s in sensors}
    ship.metrics.last_aoi_update_time = 0.0
    
    # Track actions
    action_log = []
    
    # Monkey-patch the strategy to log actions
    original_get_next_target = strategy.get_next_target
    
    def logged_get_next_target(drone, sensors, ships, config, current_time):
        result = original_get_next_target(drone, sensors, ships, config, current_time)
        
        if len(action_log) < 20:  # Only log first 20
            action_log.append({
                'time': current_time,
                'type': result.entity_type,
                'entity_id': getattr(result.entity, 'id', 'ship') if result.entity else None,
                'buffer_size': len(drone.messages),
                'buffer_full': drone.is_buffer_full(config)
            })
        
        return result
    
    strategy.get_next_target = logged_get_next_target
    
    # Run processes
    for sensor in sensors:
        env.process(sensor_process(env, sensor, config, ship.metrics))
    
    env.process(ship_process(env, ship, config))
    env.process(drone_process(env, drone, sensors, [ship], config, ship.metrics))
    
    env.run(until=duration)
    
    # Print action log
    print("\nFirst 20 decisions:")
    print(f"{'Time':>8s} | {'Type':>6s} | {'ID':>4s} | {'Buffer':>6s} | {'Full?':>5s}")
    print("-" * 50)
    
    for log in action_log:
        time_str = f"{log['time']:.0f}"
        type_str = log['type']
        id_str = str(log['entity_id']) if log['entity_id'] is not None else "N/A"
        buffer_str = str(log['buffer_size'])
        full_str = "YES" if log['buffer_full'] else "no"
        
        print(f"{time_str:>8s} | {type_str:>6s} | {id_str:>4s} | {buffer_str:>6s} | {full_str:>5s}")
    
    # Print summary
    metrics = ship.get_episode_metrics(final_time=duration)
    print(f"\nSummary:")
    print(f"  Delivered: {metrics.delivered}")
    print(f"  Delivery rate: {metrics.delivery_rate:.3f}")
    print(f"  Sensor visits: {metrics.sensor_visits_total}")
    print(f"  Ship visits: {metrics.ship_visits}")

# Test each problematic model
models = [
    ("models/dqn_no_warmup_20251216_182153/best_model.zip", "DQN No Warmup"),
    ("models/recurrent_ppo_20251216_193400/best_model.zip", "RecurrentPPO 193400"),
    ("models/recurrent_ppo_20251217_011230/best_model.zip", "RecurrentPPO 011230"),
]

for model_path, model_name in models:
    test_model_decisions(model_path, model_name)

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print("If all three models show identical action sequences,")
print("then there's a bug causing them to ignore the model predictions.")
print("If they show different actions, the issue is elsewhere.")
