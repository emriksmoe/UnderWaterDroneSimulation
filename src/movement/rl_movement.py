# src/movement/rl_movement.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

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
    Uses a trained RL model to choose next target.
    Supports: MaskablePPO, RecurrentPPO, DQN, A2C
    
    Action mapping:
        actions 0..N-1 = sensor index
        action N       = ship
    """

    model_path: str
    episode_duration: int = 86400
    deterministic: bool = True

    def __post_init__(self):
        self.model = None
        self.is_recurrent = False
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self._load_model()
        self.memory = RLMemory()

    def _load_model(self):
        """Auto-detect and load any supported model type"""
        
        # Try RecurrentPPO first
        try:
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(self.model_path)
            self.is_recurrent = True
            print(f"Loaded RecurrentPPO model from {self.model_path}")
            return
        except Exception:
            pass
        
        # Try MaskablePPO
        try:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(self.model_path)
            self.is_recurrent = False
            print(f"Loaded MaskablePPO model from {self.model_path}")
            return
        except Exception:
            pass
        
        # Try DQN
        try:
            from stable_baselines3 import DQN
            self.model = DQN.load(self.model_path)
            self.is_recurrent = False
            print(f"Loaded DQN model from {self.model_path}")
            return
        except Exception:
            pass
        
        # Try A2C
        try:
            from stable_baselines3 import A2C
            self.model = A2C.load(self.model_path)
            self.is_recurrent = False
            print(f"Loaded A2C model from {self.model_path}")
            return
        except Exception:
            pass
        
        # Try standard PPO
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(self.model_path)
            self.is_recurrent = False
            print(f"Loaded PPO model from {self.model_path}")
            return
        except Exception:
            pass
        
        raise ValueError(f"Could not load model from {self.model_path}. Tried: RecurrentPPO, MaskablePPO, DQN, A2C, PPO")

    # -------------------------
    # Reset between runs
    # -------------------------
    def reset(self):
        """Reset memory and LSTM states"""
        self.memory = RLMemory()
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)

    # -------------------------
    # Hooks called by processes.py
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

        # Edge cases
        if not sensors and ships:
            ship = ships[0]
            return TargetResult(position=ship.position, entity_type="ship", entity=ship)

        if not sensors and not ships:
            return TargetResult(position=drone.position, entity_type="none", entity=None)

        ship = ships[0] if ships else None

        # Build observation
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

        # Predict action
        if self.is_recurrent:
            # RecurrentPPO prediction with LSTM states
            action, self.lstm_states = self.model.predict(
                obs_batch,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=self.deterministic
            )
            self.episode_start = np.zeros((1,), dtype=bool)  # Not episode start anymore
        else:
            # Check if model supports action masking (MaskablePPO)
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'evaluate_actions'):
                # Create action mask (block last action + force ship if buffer full)
                action_mask = np.ones(len(sensors_ordered) + 1, dtype=bool)
                
                if hasattr(self.memory, 'last_action') and self.memory.last_action is not None:
                    action_mask[self.memory.last_action] = False
                
                # FORCE SHIP if buffer full (matches training mask)
                if drone.is_buffer_full(config):
                    action_mask[:len(sensors_ordered)] = False  # Block all sensors
                    action_mask[len(sensors_ordered)] = True    # Allow ship only
                
                try:
                    # MaskablePPO
                    action, _ = self.model.predict(
                        obs_batch, 
                        action_masks=np.array([action_mask]), 
                        deterministic=self.deterministic
                    )
                except TypeError:
                    # Model doesn't support action_masks parameter (DQN, A2C, PPO)
                    action, _ = self.model.predict(obs_batch, deterministic=self.deterministic)
            else:
                # Standard prediction (DQN, A2C, PPO)
                action, _ = self.model.predict(obs_batch, deterministic=self.deterministic)

        # Track for next decision
        self.memory.last_action = int(action)

        a = int(action)

        # Map action to target
        if a < len(sensors_ordered):
            target_sensor = sensors_ordered[a]
            return TargetResult(position=target_sensor.position, entity_type="sensor", entity=target_sensor)

        # "ship" action = last index
        if ship is not None:
            return TargetResult(position=ship.position, entity_type="ship", entity=ship)

        # fallback
        return TargetResult(position=drone.position, entity_type="none", entity=None)

    def get_strategy_name(self) -> str:
        model_type = "Unknown"
        if self.model is not None:
            model_type = type(self.model).__name__
        return f"RL ({model_type}) Movement Strategy"