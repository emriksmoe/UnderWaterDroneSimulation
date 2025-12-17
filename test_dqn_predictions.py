#!/usr/bin/env python3
"""Test what DQN no-warmup actually predicts during simulation"""

import simpy
import numpy as np
from src.config.simulation_config import SimulationConfig
from src.simulation.agent_factory import AgentFactory
from src.simulation.processes import sensor_process, ship_process
from src.movement.rl_movement import RLMovementStrategy
from analysis.metrics import MetricsCollector

def capture_dqn_decisions(model_path, num_decisions=50):
    """Capture raw DQN predictions during simulation"""
    config = SimulationConfig()
    env = simpy.Environment()
    np.random.seed(42)
    
    strategy = RLMovementStrategy(model_path=model_path, episode_duration=86400)
    factory = AgentFactory(config, drone_strategy=strategy)
    sensors, drones, ships = factory.create_all_agents()
    
    drone = drones[0]
    ship = ships[0]
    ship.metrics = MetricsCollector()
    ship.metrics.last_delivery_time_per_sensor = {s.id: 0.0 for s in sensors}
    ship.metrics.last_aoi_update_time = 0.0
    
    decisions = []
    
    original_get_next_target = strategy.get_next_target
    
    def capture_get_next_target(drone, sensors, ships, config, current_time):
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
        
        # Get RAW DQN prediction
        raw_action, _ = strategy.model.predict(obs_batch, deterministic=strategy.deterministic)
        
        result = original_get_next_target(drone, sensors, ships, config, current_time)
        
        if len(decisions) < num_decisions:
            decisions.append({
                'time': current_time,
                'raw_action': int(raw_action),
                'final_type': result.entity_type,
                'final_id': getattr(result.entity, 'id', 'ship') if result.entity else None,
                'buffer_size': len(drone.messages),
                'buffer_full': drone.is_buffer_full(config)
            })
        
        return result
    
    strategy.get_next_target = capture_get_next_target
    
    for sensor in sensors:
        env.process(sensor_process(env, sensor, config, ship.metrics))
    env.process(ship_process(env, ship, config))
    
    def simple_drone_process():
        while len(decisions) < num_decisions:
            target = strategy.get_next_target(drone, sensors, [ship], config, env.now)
            yield env.timeout(1)
    
    env.process(simple_drone_process())
    env.run(until=10000)
    
    return decisions

print("Testing DQN no-warmup predictions...")
decisions = capture_dqn_decisions("models/dqn_no_warmup_20251216_182153/best_model.zip", 30)

print("\nDQN no-warmup raw predictions:")
print(f"{'Decision':<10} | {'Raw':<6} | {'Final':<12} | {'Buffer':<8} | {'Full?':<6}")
print("-" * 55)

raw_actions = []
for i, d in enumerate(decisions, 1):
    final = f"{d['final_type']}:{d['final_id']}"
    full = "YES" if d['buffer_full'] else "no"
    print(f"{i:<10} | {d['raw_action']:<6} | {final:<12} | {d['buffer_size']:<8} | {full:<6}")
    raw_actions.append(d['raw_action'])

print("\n" + "="*55)
print(f"Unique raw actions: {len(set(raw_actions))}")
print(f"Action distribution: {dict(sorted([(a, raw_actions.count(a)) for a in set(raw_actions)], key=lambda x: x[1], reverse=True))}")
print(f"Action 20 frequency: {raw_actions.count(20)}/30 ({100*raw_actions.count(20)/30:.0f}%)")

if raw_actions.count(20) == 30:
    print("\n⚠️  DQN outputs action 20 (ship) 100% of the time!")
    print("This is the SAME degenerate behavior as RecurrentPPO.")
else:
    print(f"\n✓ DQN shows some diversity (not collapsed)")
