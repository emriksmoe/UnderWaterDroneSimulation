import numpy as np
from math import sqrt
from typing import List
from .rl_memory import RLMemory
from ..agents.sensor import Sensor
from ..agents.drone import Drone
from ..utils.position import Position
from ..agents.ship import Ship

def build_observation(drone: Drone, sensors: List[Sensor], ship: Ship, rl_memory: RLMemory, current_time: float, config, episode_duration: int) -> np.ndarray:
    """
    Build the observation vector used by both:
    - DroneAoIEnv (training)
    - RLMovementStrategy (deployment)

    Uses only allowed information:
    - positions -> distances
    - drone buffer contents
    - memory of last visits
    - time progress
    """

    area_x = config.area_size[0]
    area_y = config.area_size[1]
    depth_max = config.depth_range
    depth_min = config.min_depth
    max_xy = max(area_x, area_y)
    max_depth = max(1e-6, depth_max) 

    max_dist = sqrt(area_x**2 + area_y**2 + max_depth**2) + 1e-6

    buffer_capacity = config.drone_buffer_capacity
    num_sensors = len(sensors)
    episode_duration = episode_duration
    
    # Normalize drone position
    dx, dy, dz = drone.position.x, drone.position.y, drone.position.z
    nx = (dx / max_xy) * 2 - 1
    ny = (dy / max_xy) * 2 - 1
    nz = (dz / max_depth) * 2 - 1

    buffer_usage = len(drone.messages) / buffer_capacity

    # Distance to ship
    sx, sy, sz = ship.position.x, ship.position.y, ship.position.z
    dist_ship = sqrt((dx - sx)**2 + (dy - sy)**2 + (dz - sz)**2) / max_dist

    # Time since last ship visit
    time_since_ship = (current_time - rl_memory.last_ship_visit_time) / episode_duration
    time_since_ship = float(np.clip(time_since_ship, 0, 1))
    
    # Fraction of episode completed
    episode_frac = float(np.clip(current_time / episode_duration, 0, 1))

    # Start building observation
    obs = []
    
    # Global features
    obs.extend([nx, ny, nz])
    obs.append(buffer_usage)
    obs.append(dist_ship)
    obs.append(time_since_ship)
    obs.append(episode_frac)

    # Messages in drone by sensor ID
    msg_from_sensor = {i: 0 for i in range(num_sensors)}
    for msg in drone.messages:
        sid = getattr(msg, "source_id", None)
        if sid is not None and 0 <= sid < num_sensors:
            msg_from_sensor[sid] += 1

    # Per-sensor features
    for i, sensor in enumerate(sensors):
        sx, sy, sz = sensor.position.x, sensor.position.y, sensor.position.z

        dist = sqrt((dx - sx)**2 + (dy - sy)**2 + (dz - sz)**2) / max_dist

        carry_frac = msg_from_sensor[i] / buffer_capacity

        last_visit = rl_memory.last_sensor_visit_time.get(i, 0.0)
        time_since_visit = np.clip((current_time - last_visit) / episode_duration, 0, 1)

        last_pickup = rl_memory.last_sensor_pickup_count.get(i, 0)
        last_pickup_norm = np.clip(last_pickup / buffer_capacity, 0, 1)

        # Ship-side info about messages from this sensor
        known_gen = rl_memory.last_known_ship_gen_time.get(i, 0.0)
        time_since_known_delivery = np.clip((current_time - known_gen) / episode_duration, 0, 1)

        obs.extend([dist, carry_frac, time_since_visit, last_pickup_norm, time_since_known_delivery])

    return np.array(obs, dtype=np.float32)