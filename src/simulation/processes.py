# src/simulation/processes.py

import simpy
import random
from typing import List, Optional

from ..agents.drone import Drone
from ..agents.ship import Ship
from ..agents.sensor import Sensor
from ..config.simulation_config import SimulationConfig

from analysis.metrics import MetricsCollector


def sensor_process(
    env: simpy.Environment,
    sensor: Sensor,
    config: SimulationConfig,
    metrics: Optional[MetricsCollector] = None
):
    """Simpy process that generates sensor data at intervals."""
    while True:
        message = sensor.generate_message(env.now, config)

        # Buffer insert (may drop oldest; Sensor logs dropped via metrics)
        sensor.add_message_to_buffer(message, config, metrics)

        # Log generation
        if metrics is not None:
            metrics.log_message_generated(1)

        interval = random.expovariate(1.0 / sensor.generation_interval)
        yield env.timeout(interval)


def drone_process(
    env: simpy.Environment,
    drone: Drone,
    sensors: List[Sensor],
    ships: List[Ship],
    config: SimulationConfig,
    metrics: Optional[MetricsCollector] = None
):
    """Event-driven drone process: move → act → wait."""
    while True:
        target = drone.get_next_target(sensors, ships, config, env.now)

        travel_time = drone.calculate_travel_time(target.position, config)
        yield env.timeout(travel_time)

        drone.position = target.position

        if target.entity_type == "sensor" and target.entity is not None:
            collected = drone.collect_from_sensor(target.entity, config, env.now, metrics)
            tot_msg = collected
            
            # Log sensor visit
            if metrics is not None:
                metrics.log_sensor_visit(target.entity.id)
            
            if hasattr(drone.movement_strategy, "notify_sensor_visit"):
                drone.movement_strategy.notify_sensor_visit(target.entity.id, collected, env.now)

        elif target.entity_type == "ship" and target.entity is not None:
            delivered_count, delivered_messages = drone.deliver_to_ship(target.entity, config, env.now)
            tot_msg = delivered_count
            
            # Log ship visit
            if metrics is not None:
                metrics.log_ship_visit()
            
            if hasattr(drone.movement_strategy, "notify_ship_visit"):
                drone.movement_strategy.notify_ship_visit(env.now, delivered_messages)

        operation_time = config.communication_hand_shake_time + (tot_msg * config.communication_bitrate)
        yield env.timeout(operation_time)


def ship_process(env: simpy.Environment, ship: Ship, config: SimulationConfig):
    """SimPy process for ship operations (stationary)."""
    while True:
        yield env.timeout(60.0)