"""Single agent OpenAI Gym environment for DTN drone simulation"""

import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional
from gymnasium import spaces
import random

from src.config.simulation_config import SimulationConfig
from src.rl.state_manager import RLStateManager
from src.agents.drone import Drone
from src.agents.sensor import Sensor
from src.agents.ship import Ship
from src.utils.position import Position
from src.protocols.dtn_protocol import DTNMessage
from src.protocols.dtn_protocol import EpidemicProtocol


class DTNDroneEnvironment(gym.Env):
    """Custom OpenAI Gym environment for DTN drone simulation."""

    def __init__(self, config: SimulationConfig):
        super().__init__()

        self.config = config
        self.state_manager = RLStateManager(config)

        # State space: Must match the output of the state manager
        K = self.config.sensors_state_space
        M = self.config.ships_state_space
        state_dims = 7 + (3 * K) + (3 * M)  # Internal Drone state + Sensors state + Ships state

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(state_dims,), dtype=np.float32
        )

        # Action space: K sensors, M ships and 1 explore action
        num_actions = K + M + 1
        self.action_space = spaces.Discrete(num_actions)

        # Episode tracking
        self.current_step = 0
        self.max_steps = self.config.max_episode_steps
        self.episode_reward = 0.0
        self.current_time = 0.0

        # Mock simulation components (for standalone training)
        self.mock_drone = None
        self.mock_sensors: List[Sensor] = []
        self.mock_ships: List[Ship] = []

        # Initialize sensor timing tracking
        self.sensor_next_generation_times = {}

        # Episode statistics
        self.episode_stats = {
            'messages_delivered': 0,
            'messages_collected': 0,
            'total_distance_traveled': 0.0,
            'sensor_visits': 0,
            'ship_visits': 0,
            'explore_actions': 0
        }

        # AoI tracking
        self.aoi_metrics = {
            "delivered_messages": [],
            "collected_messages": [],
            "episode_start_time": 0.0,
            "episode_end_time": 0.0
        }

    def reset(self, seed=None, options=None):
        """Reset environment state for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # ===== STEP 1: Reset all timing FIRST =====
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_time = 0.0
        
        # ===== STEP 2: Clear old data =====
        if hasattr(self, 'mock_sensors') and self.mock_sensors:
            for sensor in self.mock_sensors:
                sensor.messages.clear()
        
        if hasattr(self, 'mock_drone') and self.mock_drone:
            self.mock_drone.messages.clear()
        
        if hasattr(self, 'mock_ships') and self.mock_ships:
            for ship in self.mock_ships:
                if hasattr(ship, 'messages'):
                    ship.messages.clear()
        
        self.sensor_next_generation_times = {}
        
        # ===== STEP 3: Reset statistics =====
        self.episode_stats = {
            'messages_delivered': 0,
            'messages_collected': 0,
            'total_distance_traveled': 0.0,
            'sensor_visits': 0,
            'ship_visits': 0,
            'explore_actions': 0
        }

        self.expired_messages = {
            'from_sensors': [],
            'from_drone': [],
            'total_penalty': 0.0
        }
        
        # Reset AoI metrics
        self.aoi_metrics = {
            "delivered_messages": [],
            "collected_messages": [],
            "episode_start_time": 0.0,
            "episode_end_time": 0.0
        }
        
        # Reset camping detection
        self.last_action = None
        self.same_action_count = 0
        self.last_target_position = None
        self.consecutive_ship_visits = 0

        # ===== STEP 4: Initialize simulation =====
        self._initialize_mock_simulation()

        # ===== STEP 5: Get initial state =====
        initial_state = self.state_manager.get_drone_state(
            self.mock_drone, self.mock_sensors, self.mock_ships, self.current_time
        )

        info = {"episode_step": 0, "simulation_time": 0.0, "stats": self.episode_stats.copy()}
        return initial_state, info

    def step(self, action):
        """Executes an action, returns (observation, reward, done, truncated, info)."""
        if self.mock_drone is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        
        # Store previous position for distance calculation
        prev_position = Position(
            self.mock_drone.position.x,
            self.mock_drone.position.y,
            self.mock_drone.position.z
        )

        # Convert action to target
        target_position, target_entity, action_type = self._action_to_target(action)

        # ✅ ENHANCED: Progressive camping detection
        camping_penalty = 0.0
        
        # Track consecutive ship visits
        if not hasattr(self, 'consecutive_ship_visits'):
            self.consecutive_ship_visits = 0
        
        # Check if we're currently at a ship or sensor
        current_at_ship = False
        current_at_sensor = False
        
        for ship in self.mock_ships:
            if prev_position.distance_to(ship.position) < 10.0:
                current_at_ship = True
                break
        
        if not current_at_ship:
            for sensor in self.mock_sensors:
                if prev_position.distance_to(sensor.position) < 10.0:
                    current_at_sensor = True
                    break
        
        # Progressive camping penalty - gets exponentially worse
        if current_at_ship and len(self.mock_drone.messages) == 0:
            self.consecutive_ship_visits += 1
            
            # Exponential penalty: gets worse each consecutive visit
            base_penalty = self.config.penalty_idle_at_ship
            progressive_multiplier = min(self.consecutive_ship_visits / 10.0, 5.0)  # Up to 5x worse
            camping_penalty += base_penalty * (1 + progressive_multiplier)
        else:
            self.consecutive_ship_visits = 0  # Reset if agent does something else
        
        # Penalize camping at sensor with full buffer
        buffer_usage = len(self.mock_drone.messages) / self.config.drone_buffer_capacity
        if current_at_sensor and buffer_usage > 0.95:
            camping_penalty += self.config.penalty_idle_at_sensor

        # Execute action and get immediate AoI rewards
        step_reward = self._execute_action(target_position, target_entity, action_type, prev_position)
        
        # Apply camping penalty
        step_reward += camping_penalty
        
        # Add continuous AoI pressure
        aoi_pressure = self._calculate_continuous_aoi_pressure()
        step_reward += aoi_pressure

        # Update simulation state
        self._update_simulation_state()

        # Get new state
        new_state = self.state_manager.get_drone_state(
            self.mock_drone, self.mock_sensors, self.mock_ships, self.current_time
        )

        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = self.current_time >= self.config.sim_time
        
        # Initialize info dictionary
        info = {
            "episode_step": self.current_step,
            "simulation_time": self.current_time,
            "total_reward": self.episode_reward,
            "action_taken": action,
            "target_type": action_type,
            "step_reward": step_reward,
            "stats": self.episode_stats.copy(),
            "expired_this_episode": len(self.expired_messages['from_sensors']) + len(self.expired_messages['from_drone']),
            "ttl_penalty_this_episode": self.expired_messages['total_penalty'],
            "consecutive_ship_visits": self.consecutive_ship_visits
        }
        
        # Episode-end handling
        if done or truncated:
            self.aoi_metrics["episode_end_time"] = self.current_time
            
            # Calculate comprehensive AoI metrics
            global_aoi_metrics = self._calculate_global_aoi_metrics()
            
            # Calculate episode-end penalty
            episode_end_penalty = self._calculate_episode_end_aoi_penalty_from_metrics(global_aoi_metrics)
            step_reward += episode_end_penalty
            
            # Add AoI metrics to info
            info.update({
                "aoi_metrics": global_aoi_metrics,
                "detailed_aoi_data": self.aoi_metrics,
                "episode_end_penalty": episode_end_penalty,
                "undelivered_messages": self._count_undelivered_messages(),
                "expired_messages": {
                    'from_sensors': len(self.expired_messages['from_sensors']),
                    'from_drone': len(self.expired_messages['from_drone']),
                    'total': len(self.expired_messages['from_sensors']) + len(self.expired_messages['from_drone']),
                    'total_penalty': self.expired_messages['total_penalty'],
                    'details': self.expired_messages
                }
            })
        
        # Update episode reward
        self.episode_reward += step_reward
        
        return new_state, step_reward, done, truncated, info

    def _action_to_target(self, action: int) -> Tuple[Position, Optional[object], str]:
        """Convert RL action to target position and entity and action type."""
        K = self.config.sensors_state_space
        M = self.config.ships_state_space

        if action < K:  # Choose specific sensor
            sensor_idx = action

            if self.mock_sensors:
                # Sorts by distance to be same as state manager
                sensor_distances = [(sensor, self.mock_drone.position.distance_to(sensor.position)) 
                                   for sensor in self.mock_sensors]
                sensor_distances.sort(key=lambda x: x[1])
                
                if sensor_idx < len(sensor_distances):
                    target_sensor = sensor_distances[sensor_idx][0]
                    return target_sensor.position, target_sensor, 'sensor'
                
        elif action < K + M:  # Choose specific ship
            ship_idx = action - K

            if self.mock_ships:
                # Sorts by distance to be same as state manager
                ship_distances = [(ship, self.mock_drone.position.distance_to(ship.position)) 
                                   for ship in self.mock_ships]
                ship_distances.sort(key=lambda x: x[1])
                
                if ship_idx < len(ship_distances):
                    target_ship = ship_distances[ship_idx][0]
                    return target_ship.position, target_ship, 'ship'
                
        # Explore action: random nearby position
        random_x = random.uniform(0, self.config.area_size[0])
        random_y = random.uniform(0, self.config.area_size[1])
        random_z = random.uniform(self.config.min_depth, self.config.depth_range)

        return Position(random_x, random_y, random_z), None, "explore"
    
    def _execute_action(self, target_position: Position, target_entity: Optional[object], action_type: str, prev_position: Position) -> float:
        """Execute the given action and return the reward."""
        
        travel_distance = prev_position.distance_to(target_position)
        travel_time = travel_distance / self.config.drone_speed
        
        # Cap travel time to prevent excessive time advancement
        travel_time = min(travel_time, 60.0)
        
        # Move drone to target
        self.mock_drone.position = target_position
        self.episode_stats['total_distance_traveled'] += travel_distance
        self.current_time += travel_time
        
        reward = 0.0

        buffer_usage = len(self.mock_drone.messages) / self.config.drone_buffer_capacity
    
        # If buffer is critically full and NOT going to ship, heavy penalty
        if buffer_usage > 0.8 and action_type != "ship":
            reward += self.config.penalty_buffer_near_full
        
        # Sensor interaction
        if action_type == "sensor" and target_entity:
            self.episode_stats['sensor_visits'] += 1

            available_space = self.config.drone_buffer_capacity - len(self.mock_drone.messages)
            
            if available_space == 0:
                # Buffer full - should not be at sensor!
                reward += self.config.penalty_buffer_overflow / 2
            elif target_entity.has_messages():
                collected_reward = 0.0
                messages_collected = 0
                
                while (available_space > 0 and target_entity.has_messages()):
                    message = target_entity.get_next_message_for_collection()
                    if message:
                        collection_aoi = self._validate_aoi(
                            message.generation_time,
                            self.current_time,
                            message.id
                        )
                        age_ratio = min(collection_aoi / self.config.message_ttl, 1.0)
                        
                        self.aoi_metrics["collected_messages"].append({
                            "message_id": message.id,
                            "collection_time": self.current_time,
                            "generation_time": message.generation_time,
                            "collection_aoi": collection_aoi,
                            "sensor_id": target_entity.id
                        })
                        
                        urgency_reward = (
                            self.config.reward_collection_base + 
                            (age_ratio * self.config.reward_collection_urgency_multiplier)
                        )
                        collected_reward += urgency_reward
                        
                        self.mock_drone.messages.append(message)
                        messages_collected += 1
                        available_space -= 1
                        
                        # Check if buffer just became full
                        new_buffer_usage = len(self.mock_drone.messages) / self.config.drone_buffer_capacity
                        if new_buffer_usage >= 1.0:
                            break
                    else:
                        break
                
                if messages_collected > 0:
                    self.episode_stats['messages_collected'] += messages_collected
                    reward += collected_reward
            else:
                reward += self.config.penalty_empty_sensor
        
        # Ship interaction  
        elif action_type == "ship" and target_entity:
            self.episode_stats['ship_visits'] += 1

            if len(self.mock_drone.messages) > 0:
                delivery_reward = 0.0
                messages_to_deliver = self.mock_drone.messages.copy()
                messages_delivered = len(messages_to_deliver)

                for message in messages_to_deliver:
                    delivery_aoi = self._validate_aoi(
                        message.generation_time,
                        self.current_time,
                        message.id
                    )
                    age_ratio = min(delivery_aoi / self.config.message_ttl, 1.0)
                    freshness_ratio = 1.0 - age_ratio
                    
                    # Transfer message to ship
                    target_entity.receive_message(message, self.current_time)
                    
                    # Track for metrics
                    self.aoi_metrics["delivered_messages"].append({
                        "message_id": message.id,
                        "delivery_time": self.current_time,
                        "generation_time": message.generation_time,
                        "delivery_aoi": delivery_aoi,
                        "ship_id": target_entity.id
                    })

                    freshness_reward = (
                        self.config.reward_delivery_base + 
                        (freshness_ratio * self.config.reward_delivery_freshness_multiplier)
                    )
                    delivery_reward += freshness_reward
                
                # Clear drone buffer after delivery
                self.mock_drone.messages.clear()
                
                self.episode_stats['messages_delivered'] += messages_delivered
                reward += delivery_reward
                
                # Bonus for delivering when buffer was nearly full
                if buffer_usage > 0.9:
                    buffer_relief_bonus = 100.0
                    reward += buffer_relief_bonus
            else:
                reward += self.config.penalty_ship_no_messages
        
        # Explore action
        elif action_type == "explore":
            self.episode_stats['explore_actions'] += 1
            reward += self.config.penalty_explore
            
            # Extra penalty if exploring with full buffer
            if buffer_usage > 0.8:
                reward += self.config.penalty_buffer_near_full
        
        # Movement penalty
        reward += travel_time * self.config.penalty_time_per_second
        
        # Carrying penalty (progressive with age)
        for message in self.mock_drone.messages:
            message_age = max(0.0, self.current_time - message.generation_time)
            reward += message_age * self.config.penalty_carrying_per_age_unit
        
        # Buffer overflow check
        final_buffer_usage = len(self.mock_drone.messages) / self.config.drone_buffer_capacity
        if final_buffer_usage > 1.0:
            reward += self.config.penalty_buffer_overflow
        
        return reward
    
    def _validate_aoi(self, message_time: float, current_time: float, message_id: str = "unknown") -> float:
        """Validate AoI calculation and prevent negative values"""
        aoi = current_time - message_time
        
        if aoi < 0:
            aoi = 0.0
        
        return aoi
    
    def _calculate_global_aoi_metrics(self) -> dict:
        """Calculate comprehensive AoI metrics at episode end"""
        
        # Delivered message AoI statistics
        delivered_aois = [msg["delivery_aoi"] for msg in self.aoi_metrics["delivered_messages"]]
        
        # Undelivered message AoI (current age)
        undelivered_aois = []
        current_time = self.current_time
        
        # Messages still in sensors
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                current_aoi = max(0.0, current_time - message.generation_time)
                undelivered_aois.append({
                    "message_id": message.id,
                    "current_aoi": current_aoi,
                    "location": "sensor",
                    "entity_id": sensor.id
                })
        
        # Messages still in drone buffer
        for message in self.mock_drone.messages:
            current_aoi = max(0.0, current_time - message.generation_time)
            undelivered_aois.append({
                "message_id": message.id,
                "current_aoi": current_aoi,
                "location": "drone",
                "entity_id": self.mock_drone.id
            })
        
        # Calculate statistics
        undelivered_aoi_sum = sum(msg["current_aoi"] for msg in undelivered_aois)
        delivered_aoi_sum = sum(delivered_aois)
        
        metrics = {
            "delivered": {
                "count": len(delivered_aois),
                "mean_aoi": delivered_aoi_sum / len(delivered_aois) if delivered_aois else 0.0,
                "min_aoi": min(delivered_aois) if delivered_aois else 0.0,
                "max_aoi": max(delivered_aois) if delivered_aois else 0.0
            },
            "undelivered": {
                "count": len(undelivered_aois),
                "total_current_aoi": undelivered_aoi_sum,
                "mean_current_aoi": undelivered_aoi_sum / len(undelivered_aois) if undelivered_aois else 0.0,
                "in_sensors": sum(1 for msg in undelivered_aois if msg["location"] == "sensor"),
                "in_drone": sum(1 for msg in undelivered_aois if msg["location"] == "drone")
            },
            "global": {
                "total_messages": len(delivered_aois) + len(undelivered_aois),
                "delivery_rate": len(delivered_aois) / (len(delivered_aois) + len(undelivered_aois)) if (len(delivered_aois) + len(undelivered_aois)) > 0 else 0.0,
                "system_aoi": delivered_aoi_sum + undelivered_aoi_sum
            }
        }
        
        return metrics
    
    def _calculate_episode_end_aoi_penalty_from_metrics(self, global_aoi_metrics: dict) -> float:
        """Calculate progressive episode-end penalty based on actual AoI values"""
        total_penalty = 0.0
        
        # Progressive penalties for messages still in sensors (never collected)
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                message_age = self._validate_aoi(
                    message.generation_time,
                    self.current_time,
                    message.id
                )
                
                age_penalty = (
                    self.config.penalty_undelivered_base + 
                    (message_age * self.config.penalty_undelivered_age_multiplier)
                )
                # Extra harsh penalty for never being collected
                uncollected_penalty = age_penalty * abs(self.config.penalty_uncollected_multiplier)
                total_penalty += uncollected_penalty
        
        # Progressive penalties for messages in drone buffer (collected but not delivered)
        for message in self.mock_drone.messages:
            message_age = self._validate_aoi(
                message.generation_time,
                self.current_time,
                message.id
            )
            
            age_penalty = (
                self.config.penalty_undelivered_base + 
                (message_age * self.config.penalty_undelivered_age_multiplier)
            )
            total_penalty += age_penalty
        
        return total_penalty

    def _calculate_continuous_aoi_pressure(self) -> float:
        """Enhanced continuous pressure - penalize uncollected messages heavily"""
        
        total_aoi = 0.0
        message_count = 0
        uncollected_count = 0
        current_time = self.current_time
        
        # Calculate system AoI + count uncollected
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                age = max(0.0, current_time - message.generation_time)
                total_aoi += age
                message_count += 1
                uncollected_count += 1
        
        for message in self.mock_drone.messages:
            age = max(0.0, current_time - message.generation_time)
            total_aoi += age
            message_count += 1
        
        if message_count == 0:
            return 0.0
        
        # Strong AoI pressure
        mean_aoi = total_aoi / message_count
        aoi_penalty = -(mean_aoi * 0.01)  # 10x stronger than before
        
        # HUGE penalty for uncollected messages
        uncollected_penalty = -(uncollected_count * 1.0)  # -1 per uncollected per step
        
        return aoi_penalty + uncollected_penalty

    def _count_undelivered_messages(self) -> dict:
        """Count undelivered messages for episode stats"""
        sensor_messages = sum(len(sensor.messages) for sensor in self.mock_sensors)
        drone_messages = len(self.mock_drone.messages)
        
        return {
            "in_sensors": sensor_messages,
            "in_drone": drone_messages,
            "total": sensor_messages + drone_messages
        }
        
    def _initialize_mock_simulation(self):
        """Initialize mock simulation components"""
        
        if self.current_time < 0:
            self.current_time = 0.0
        
        # Create mock drone at random position
        start_x = random.uniform(0, self.config.area_size[0])
        start_y = random.uniform(0, self.config.area_size[1])
        start_z = random.uniform(self.config.min_depth, self.config.depth_range)
        
        self.mock_drone = Drone(
            id="training_drone",
            position=Position(start_x, start_y, start_z),
            protocol=EpidemicProtocol("training_drone"),
            movement_strategy=None
        )
        
        # Create sensors
        self.mock_sensors = []
        total_sensors = max(self.config.num_sensors, 8)
        
        self.sensor_next_generation_times = {}
        safe_interval = min(self.config.data_generation_interval, 120.0)
        
        for i in range(total_sensors):
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            sensor = Sensor(id=f'mock_sensor_{i}', position=Position(x, y, z))
            
            # Schedule next message generation
            self.sensor_next_generation_times[sensor.id] = (
                self.current_time + random.uniform(10.0, safe_interval)
            )
            
            # Add initial messages with past timestamps
            if random.random() < 0.6:
                num_initial = random.randint(1, 3)
                for j in range(num_initial):
                    if self.current_time <= 1.0:
                        intended_gen_time = 0.001 * (j + 1)
                    else:
                        seconds_ago = (j + 1) * 30.0
                        intended_gen_time = self.current_time - seconds_ago
                        
                        if intended_gen_time <= 0:
                            intended_gen_time = 0.001 * (j + 1)
                    
                    msg = DTNMessage(
                        id=f'{sensor.id}_init_{j}',
                        source_id=sensor.id,
                        destination_id="surface_gateway",
                        data=f"initial_data_{j}",
                        generation_time=intended_gen_time,
                        hop_count=0,
                        priority=1,
                        ttl=self.config.message_ttl,
                        size=self.config.message_size
                    )
                    sensor.messages.append(msg)
                
                if hasattr(sensor, 'data_sequence'):
                    sensor.data_sequence = num_initial
            
            self.mock_sensors.append(sensor)
        
        # Create ships
        self.mock_ships = []
        total_ships = max(self.config.num_ships, 3)
        
        for i in range(total_ships):
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            
            ship = Ship(id=f'mock_ship_{i}', position=Position(x, y, 0.0))
            self.mock_ships.append(ship)

    def _update_simulation_state(self):
        """Enhanced state update with TTL expiration penalties"""
        
        time_delta = self.current_time - getattr(self, 'last_sim_update', 0)
        self.last_sim_update = self.current_time
        
        safe_interval = min(self.config.data_generation_interval, 120.0)
        
        # Generate messages for each sensor
        for sensor in self.mock_sensors:
            if sensor.id in self.sensor_next_generation_times:
                
                next_gen = self.sensor_next_generation_times[sensor.id]
                
                if (next_gen > self.current_time + safe_interval * 2 or 
                    next_gen < self.current_time - 300.0):
                    self.sensor_next_generation_times[sensor.id] = (
                        self.current_time + random.uniform(10.0, safe_interval)
                    )
                
                generation_count = 0
                while (self.sensor_next_generation_times[sensor.id] <= self.current_time and 
                    generation_count < 5):
                    
                    if len(sensor.messages) >= self.config.sensor_buffer_capacity:
                        sensor.messages.pop(0)
                    
                    scheduled_gen_time = self.sensor_next_generation_times[sensor.id]
                    
                    new_msg = sensor.generate_message(
                        current_time=scheduled_gen_time,
                        config=self.config
                    )
                    
                    if new_msg.generation_time > self.current_time:
                        new_msg.generation_time = max(0.001, self.current_time - random.uniform(1.0, 10.0))
                    
                    new_msg.generation_time = max(0.001, new_msg.generation_time)
                    
                    sensor.messages.append(new_msg)
                    
                    self.sensor_next_generation_times[sensor.id] = (
                        self.current_time + safe_interval
                    )
                    generation_count += 1
        
        # TTL cleanup with penalties
        if not hasattr(self, 'last_ttl_check'):
            self.last_ttl_check = 0
        
        if self.current_time - self.last_ttl_check >= self.config.message_ttl_check_interval:
            current_time = self.current_time
            ttl_penalty = 0.0
            
            # Check sensors for expired messages
            for sensor in self.mock_sensors:
                expired_in_sensor = []
                valid_messages = []
                
                for msg in sensor.messages:
                    age = current_time - msg.generation_time
                    if age > msg.ttl:
                        expired_in_sensor.append({
                            'message_id': msg.id,
                            'age': age,
                            'location': 'sensor',
                            'sensor_id': sensor.id
                        })
                        ttl_penalty += self.config.penalty_message_expired_at_sensor
                    else:
                        valid_messages.append(msg)
                
                sensor.messages = valid_messages
                
                if expired_in_sensor:
                    self.expired_messages['from_sensors'].extend(expired_in_sensor)
            
            # Check drone buffer for expired messages
            expired_in_drone = []
            valid_drone_messages = []
            
            for msg in self.mock_drone.messages:
                age = current_time - msg.generation_time
                if age > msg.ttl:
                    expired_in_drone.append({
                        'message_id': msg.id,
                        'age': age,
                        'location': 'drone'
                    })
                    ttl_penalty += self.config.penalty_message_expired
                else:
                    valid_drone_messages.append(msg)
            
            self.mock_drone.messages = valid_drone_messages
            
            if expired_in_drone:
                self.expired_messages['from_drone'].extend(expired_in_drone)
            
            # Apply TTL penalty
            if ttl_penalty < 0:
                self.episode_reward += ttl_penalty
                self.expired_messages['total_penalty'] += ttl_penalty
            
            self.last_ttl_check = current_time

    def close(self):
        """Clean up environment"""
        pass