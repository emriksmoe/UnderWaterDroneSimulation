#!/usr/bin/env python3
"""Compare first 50 model decisions during ACTUAL simulation (not random obs)"""

import simpy
import numpy as np
from src.config.simulation_config import SimulationConfig
from src.simulation.agent_factory import AgentFactory
from src.simulation.processes import sensor_process, ship_process
from src.movement.rl_movement import RLMovementStrategy
from analysis.metrics import MetricsCollector

def capture_decisions(model_path, model_name, num_decisions=50):
    """Run simulation and capture model decisions"""
    config = SimulationConfig()
    env = simpy.Environment()
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create strategy
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
    
    # Capture decisions
    decisions = []
    
    # Monkey-patch get_next_target to capture raw model predictions
    original_get_next_target = strategy.get_next_target
    
    def capture_get_next_target(drone, sensors, ships, config, current_time):
        # Build observation
        sensors_ordered = sorted(sensors, key=lambda s: s.id)
        from src.rl.state_builder import build_observation
        obs = build_observation(
            drone=drone,
            sensors=sensors_ordered,
            ship=ships[0] if ships else None,
            current_time=float(current_time),
            config=config,
            rl_memory=strategy.memory,
            episode_duration=strategy.episode_duration,
        )
        
        obs_batch = np.expand_dims(obs, axis=0)
        
        # Get RAW model prediction
        if strategy.is_recurrent:
            raw_action, strategy.lstm_states = strategy.model.predict(
                obs_batch,
                state=strategy.lstm_states,
                episode_start=strategy.episode_start,
                deterministic=strategy.deterministic
            )
            strategy.episode_start = np.zeros((1,), dtype=bool)
        else:
            raw_action, _ = strategy.model.predict(obs_batch, deterministic=strategy.deterministic)
        
        # Now call original to get final action (with enforcements)
        result = original_get_next_target(drone, sensors, ships, config, current_time)
        
        if len(decisions) < num_decisions:
            decisions.append({
                'time': current_time,
                'raw_action': int(raw_action) if hasattr(raw_action, '__iter__') else int(raw_action),
                'final_type': result.entity_type,
                'final_id': getattr(result.entity, 'id', 'ship') if result.entity else None,
                'buffer_size': len(drone.messages),
                'buffer_full': drone.is_buffer_full(config)
            })
        
        return result
    
    strategy.get_next_target = capture_get_next_target
    
    # Run processes
    for sensor in sensors:
        env.process(sensor_process(env, sensor, config, ship.metrics))
    
    env.process(ship_process(env, ship, config))
    
    # Simple drone process just for testing
    def simple_drone_process():
        while len(decisions) < num_decisions:
            target = strategy.get_next_target(drone, sensors, [ship], config, env.now)
            if target.entity_type == "sensor":
                yield env.timeout(1)  # Simulate movement
            elif target.entity_type == "ship":
                yield env.timeout(1)
            else:
                break
    
    env.process(simple_drone_process())
    env.run(until=10000)  # Should be enough
    
    return decisions

# Test both RecurrentPPO models
models = [
    ("models/recurrent_ppo_20251216_193400/best_model.zip", "RP1 (193400)"),
    ("models/recurrent_ppo_20251217_011230/best_model.zip", "RP2 (011230)"),
]

results = {}
for model_path, name in models:
    print(f"Testing {name}...")
    decisions = capture_decisions(model_path, name, num_decisions=30)
    results[name] = decisions

# Compare
print("\n" + "="*80)
print("COMPARISON OF RAW MODEL PREDICTIONS (first 30 decisions)")
print("="*80)

print(f"\n{'Decision':<10} | {'RP1 Raw':<10} | {'RP2 Raw':<10} | {'Same?':<6} | {'RP1 Final':<12} | {'RP2 Final':<12}")
print("-" * 80)

rp1_decisions = results["RP1 (193400)"]
rp2_decisions = results["RP2 (011230)"]

identical_raw = 0
identical_final = 0

for i in range(min(30, len(rp1_decisions), len(rp2_decisions))):
    d1 = rp1_decisions[i]
    d2 = rp2_decisions[i]
    
    same_raw = d1['raw_action'] == d2['raw_action']
    same_final = (d1['final_type'], d1['final_id']) == (d2['final_type'], d2['final_id'])
    
    if same_raw:
        identical_raw += 1
    if same_final:
        identical_final += 1
    
    final1 = f"{d1['final_type']}:{d1['final_id']}"
    final2 = f"{d2['final_type']}:{d2['final_id']}"
    
    print(f"{i+1:<10} | {d1['raw_action']:<10} | {d2['raw_action']:<10} | {'YES' if same_raw else 'no':<6} | {final1:<12} | {final2:<12}")

print("\n" + "="*80)
print(f"Raw action identity: {identical_raw}/30 ({100*identical_raw/30:.0f}%)")
print(f"Final action identity: {identical_final}/30 ({100*identical_final/30:.0f}%)")

if identical_raw == 30:
    print("\n⚠️  CRITICAL: Models produce IDENTICAL raw predictions even on real observations!")
    print("This confirms both models learned the same degenerate policy.")
else:
    print(f"\n✓ Models diverge: They produce different predictions {30-identical_raw}/30 times")

# Action distribution
rp1_raw_actions = [d['raw_action'] for d in rp1_decisions]
rp2_raw_actions = [d['raw_action'] for d in rp2_decisions]

print(f"\nRP1 unique raw actions: {len(set(rp1_raw_actions))}")
print(f"RP1 action distribution: {dict(sorted([(a, rp1_raw_actions.count(a)) for a in set(rp1_raw_actions)], key=lambda x: x[1], reverse=True))}")

print(f"\nRP2 unique raw actions: {len(set(rp2_raw_actions))}")
print(f"RP2 action distribution: {dict(sorted([(a, rp2_raw_actions.count(a)) for a in set(rp2_raw_actions)], key=lambda x: x[1], reverse=True))}")
