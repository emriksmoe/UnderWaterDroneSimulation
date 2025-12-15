# src/movement/rl_movement.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from .movement_strategy import MovementStrategy, TargetResult
from ..config.simulation_config import SimulationConfig
from ..agents.drone import Drone
from ..agents.sensor import Sensor
from ..agents.ship import Ship

from ..rl.rl_memory import RLMemory
from ..rl.state_builder import build_observation


@dataclass
class RLMovementStrategy(MovementStrategy):
    """
    Uses a trained Stable-Baselines3 PPO model to choose next target.
    Action mapping must match training env:
        actions 0..N-1 = sensor index
        action N       = ship
    """

    model_path: str
    episode_duration: int = 86400
    deterministic: bool = True

    def __post_init__(self):
        self.model = MaskablePPO.load(self.model_path)
        self.memory = RLMemory()

    # -------------------------
    # Optional: reset between runs
    # -------------------------
    def reset(self):
        self.memory.reset()

    # -------------------------
    # Hooks called by processes.py (recommended)
    # -------------------------
    def notify_sensor_visit(self, sensor_id: int, pickup_count: int, now: float):
        self.memory.last_sensor_visit_time[int(sensor_id)] = float(now)
        self.memory.last_sensor_pickup_count[int(sensor_id)] = int(pickup_count)

    def notify_ship_visit(self, now: float, delivered_msgs: list):
        self.memory.last_ship_visit_time = float(now)
        for msg in delivered_msgs:
            sid = int(getattr(msg, "source_id", -1))
            if sid >= 0:
                prev = self.memory.last_known_ship_gen_time.get(sid, 0.0)
                self.memory.last_known_ship_gen_time[sid] = max(prev, float(msg.generation_time))


    # -------------------------
    # Main decision function
    # -------------------------
    def get_next_target(
        self,
        drone: Drone,
        sensors: List[Sensor],
        ships: List[Ship],
        config: SimulationConfig,
        current_time: float,
    ) -> TargetResult:

        if not sensors and ships:
            ship = ships[0]
            return TargetResult(position=ship.position, entity_type="ship", entity=ship)

        if not sensors and not ships:
            return TargetResult(position=drone.position, entity_type="none", entity=None)

        ship = ships[0] if ships else None

        sensors_ordered = sorted(sensors, key=lambda s: s.id)
        obs = build_observation(
            drone=drone,
            sensors=sensors_ordered,
            ship=ship,
            current_time=float(current_time),
            config=config,
            rl_memory=self.memory,
            episode_duration=int(self.episode_duration),
        )

        obs_batch = np.expand_dims(obs, axis=0)

        # Create action mask (block last action)
        action_mask = np.ones(len(sensors_ordered) + 1, dtype=bool)
        if hasattr(self.memory, 'last_action') and self.memory.last_action is not None:
            action_mask[self.memory.last_action] = False

        # ADD THIS BLOCK HERE ⬇️
        # FORCE SHIP if buffer full (matches training mask)
        if drone.is_buffer_full(config):
            action_mask[:len(sensors_ordered)] = False  # Block all sensors
            action_mask[len(sensors_ordered)] = True    # Allow ship only

        # MaskablePPO needs action_masks parameter (batch dimension)
        action, _ = self.model.predict(obs_batch, action_masks=np.array([action_mask]), deterministic=self.deterministic)

        # Track for next decision
        self.memory.last_action = int(action)

        a = int(action)

        if a < len(sensors_ordered):
            target_sensor = sensors_ordered[a]
            return TargetResult(position=target_sensor.position, entity_type="sensor", entity=target_sensor)

        # "ship" action = last index
        if ship is not None:
            return TargetResult(position=ship.position, entity_type="ship", entity=ship)

        # fallback
        return TargetResult(position=drone.position, entity_type="none", entity=None)

    def get_strategy_name(self) -> str:
        return "RL (PPO) Movement Strategy"
