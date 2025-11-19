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
            low=-1.0, high=1.0, shape=(state_dims,), dtype=np.float32
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

        #Initialize sensor timing tracking
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

        #AoI tracking
        self.aoi_metrics = {
            "delivered_messages": [],       # List of delivered message AoI values
            "collected_messages": [],       # List of AoI when messages were collected
            "episode_start_time": 0.0,
            "episode_end_time": 0.0
        }

    def reset(self, seed=None, options=None):
        """Reset environment state for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sensor_next_generation_times = {} # Reset sensor timing tracking

        self._initialize_mock_simulation()

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_time = 0.0
        self.episode_stats = {
            'messages_delivered': 0,
            'messages_collected': 0,
            'total_distance_traveled': 0.0,
            'sensor_visits': 0,
            'ship_visits': 0,
            'explore_actions': 0
        }
        # Reset AoI tracking
        self.aoi_metrics = {
            "delivered_messages": [],
            "collected_messages": [],
            "episode_start_time": self.current_time,
            "episode_end_time": 0.0
        }

        # Get initial state
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

        # Execute action and get immediate AoI rewards
        step_reward = self._execute_action(target_position, target_entity, action_type, prev_position)
        
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
        
        # Initialize info dictionary FIRST
        info = {
            "episode_step": self.current_step,
            "simulation_time": self.current_time,
            "total_reward": self.episode_reward,
            "action_taken": action,
            "target_type": action_type,
            "step_reward": step_reward,
            "stats": self.episode_stats.copy()
        }
        
        # Episode-end AoI penalty for undelivered messages
        if done or truncated:
            self.aoi_metrics["episode_end_time"] = self.current_time
            
            # Calculate comprehensive AoI metrics
            global_aoi_metrics = self._calculate_global_aoi_metrics()
            
            # Calculate episode-end penalty
            episode_end_penalty = self._calculate_episode_end_aoi_penalty_from_metrics(global_aoi_metrics)
            step_reward += episode_end_penalty
            
            # Add AoI metrics to info (NOW info is defined)
            info.update({
                "aoi_metrics": global_aoi_metrics,
                "detailed_aoi_data": self.aoi_metrics,
                "episode_end_penalty": episode_end_penalty,
                "undelivered_messages": self._count_undelivered_messages()
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
    
    def _execute_action(self, target_position: Position, target_entity: Optional[object], 
                       action_type: str, prev_position: Position) -> float:
        """Execute the given action and return the reward."""
        
        travel_distance = prev_position.distance_to(target_position)
        travel_time = travel_distance / self.config.drone_speed
        
        # Move drone to target
        self.mock_drone.position = target_position
        self.episode_stats['total_distance_traveled'] += travel_distance
        self.current_time += travel_time
        
        reward = 0.0
        
        # Sensor interaction
        if action_type == "sensor" and target_entity:
            self.episode_stats['sensor_visits'] += 1
            
            if target_entity.has_messages():
                collected_reward = 0.0
                # Collect messages
                available_space = self.config.drone_buffer_capacity - len(self.mock_drone.messages)
                messages_collected = 0
                
                while (available_space > 0 and target_entity.has_messages()):
                    message = target_entity.get_next_message_for_collection()
                    if message:
                        #Collecting AoI tracking
                        collection_aoi = self.current_time - message.generation_time
                        age_ratio = min(collection_aoi / self.config.message_ttl, 1.0)  # Cap at 1.0
                        self.aoi_metrics["collected_messages"].append({
                            "message_id": message.id,
                            "collection_time": self.current_time,
                            "generation_time": message.generation_time,
                            "collection_aoi": collection_aoi,
                            "sensor_id": target_entity.id
                        })
                    # PROGRESSIVE collection reward based on urgency
                        urgency_reward = (
                            self.config.reward_collection_base + 
                            (age_ratio * self.config.reward_collection_urgency_multiplier)
                        )
                        collected_reward += urgency_reward
                        

                        self.mock_drone.messages.append(message)
                        messages_collected += 1
                        available_space -= 1
                    else:
                        break
                
                if messages_collected > 0:
                    self.episode_stats['messages_collected'] += messages_collected
                    reward += collected_reward
            else:
                # Penalty for visiting empty sensor
                reward += self.config.penalty_empty_sensor
        
        # Ship interaction  
        elif action_type == "ship" and target_entity:
            self.episode_stats['ship_visits'] += 1

            if len(self.mock_drone.messages) > 0: #If there are messages to deliver
                delivery_reward = 0.0
                messages_delivered = len(self.mock_drone.messages)

                # Track AoI for each delivered message
                for message in self.mock_drone.messages:
                    delivery_aoi = self.current_time - message.generation_time
                    age_ratio = min(delivery_aoi / self.config.message_ttl, 1.0)  # Cap at 1.0
                    freshness_ratio = 1.0 - age_ratio
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
                            
                # Deliver all messages
                for message in self.mock_drone.messages:
                    target_entity.receive_message(message, self.current_time)
                
                self.mock_drone.messages.clear()
                
                self.episode_stats['messages_delivered'] += messages_delivered
                reward += delivery_reward
            else:
                # Penalty for visiting ship with no messages
                reward += self.config.penalty_ship_no_messages
        
        # Explore action
        elif action_type == "explore":
            self.episode_stats['explore_actions'] += 1
            reward += self.config.penalty_explore
        
        # Movement penalty
        reward += travel_time * self.config.penalty_time_per_second
        
        for message in self.mock_drone.messages:
            message_age = self.current_time - message.generation_time
            reward += message_age * self.config.penalty_carrying_per_age_unit
        # Buffer management
        buffer_usage = len(self.mock_drone.messages) / self.config.drone_buffer_capacity
        if buffer_usage >= 0.95:
            reward += self.config.penalty_buffer_overflow
        
        return reward
    
    def _calculate_global_aoi_metrics(self) -> dict:
        """Calculate comprehensive AoI metrics at episode end"""
        
        # 1. Delivered message AoI statistics
        delivered_aois = [msg["delivery_aoi"] for msg in self.aoi_metrics["delivered_messages"]]
        
        # 2. Undelivered message AoI (current age)
        undelivered_aois = []
        
        # Messages still in sensors
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                current_aoi = self.current_time - message.generation_time
                undelivered_aois.append({
                    "message_id": message.id,
                    "current_aoi": current_aoi,
                    "location": "sensor",
                    "entity_id": sensor.id
                })
        
        # Messages still in drone buffer
        for message in self.mock_drone.messages:
            current_aoi = self.current_time - message.generation_time
            undelivered_aois.append({
                "message_id": message.id,
                "current_aoi": current_aoi,
                "location": "drone",
                "entity_id": self.mock_drone.id
            })
        
        # Calculate statistics
        metrics = {
            "delivered": {
                "count": len(delivered_aois),
                "mean_aoi": sum(delivered_aois) / len(delivered_aois) if delivered_aois else 0.0,
                "min_aoi": min(delivered_aois) if delivered_aois else 0.0,
                "max_aoi": max(delivered_aois) if delivered_aois else 0.0
            },
            "undelivered": {
                "count": len(undelivered_aois),
                "total_current_aoi": sum(msg["current_aoi"] for msg in undelivered_aois),
                "mean_current_aoi": sum(msg["current_aoi"] for msg in undelivered_aois) / len(undelivered_aois) if undelivered_aois else 0.0,
                "in_sensors": len([msg for msg in undelivered_aois if msg["location"] == "sensor"]),
                "in_drone": len([msg for msg in undelivered_aois if msg["location"] == "drone"])
            },
            "global": {
                "total_messages": len(delivered_aois) + len(undelivered_aois),
                "delivery_rate": len(delivered_aois) / (len(delivered_aois) + len(undelivered_aois)) if (len(delivered_aois) + len(undelivered_aois)) > 0 else 0.0,
                "system_aoi": sum(delivered_aois) + sum(msg["current_aoi"] for msg in undelivered_aois)
            }
        }
        
        return metrics
    
    def _calculate_episode_end_aoi_penalty_from_metrics(self, global_aoi_metrics: dict) -> float:
        """Calculate progressive episode-end penalty based on actual AoI values"""
        total_penalty = 0.0
        
        # Progressive penalties for messages still in sensors (never collected)
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                message_age = self.current_time - message.generation_time
                
                # Progressive penalty based on actual age
                age_penalty = (
                    self.config.penalty_undelivered_base + 
                    (message_age * self.config.penalty_undelivered_age_multiplier)
                )
                
                # Extra penalty multiplier for never being collected
                uncollected_penalty = age_penalty * self.config.penalty_uncollected_multiplier
                total_penalty += uncollected_penalty
        
        # Progressive penalties for messages in drone buffer (collected but not delivered)
        for message in self.mock_drone.messages:
            message_age = self.current_time - message.generation_time
            
            # Progressive penalty based on actual age (less severe than uncollected)
            age_penalty = (
                self.config.penalty_undelivered_base + 
                (message_age * self.config.penalty_undelivered_age_multiplier)
            )
            total_penalty += age_penalty
        
        return total_penalty
    

    def _calculate_continuous_aoi_pressure(self) -> float:
        """Small continuous penalty proportional to system-wide AoI"""
        total_aoi = 0.0
        message_count = 0
        
        # Calculate total system AoI for all undelivered messages
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                message_age = self.current_time - message.generation_time
                total_aoi += message_age
                message_count += 1
        
        for message in self.mock_drone.messages:
            message_age = self.current_time - message.generation_time
            total_aoi += message_age
            message_count += 1
        
        if message_count > 0:
            # Small penalty proportional to mean system AoI
            mean_aoi = total_aoi / message_count
            return -mean_aoi * 0.001  # Small continuous pressure (negative penalty)
        return 0.0

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
        """Initialize mock simulation components with realistic entity counts"""
        # Create mock drone at random position
        start_x = random.uniform(0, self.config.area_size[0])
        start_y = random.uniform(0, self.config.area_size[1])
        start_z = random.uniform(self.config.min_depth, self.config.depth_range)
        
        self.mock_drone = Drone(
            id="training_drone",
            position=Position(start_x, start_y, start_z),
            protocol=EpidemicProtocol("training_drone"),
            movement_strategy=None  # No movement strategy needed for mock
        )
        
        # Create realistic number of sensors (more than state space)
        self.mock_sensors = []
        total_sensors = max(self.config.num_sensors, 8)
        
        # Track when each sensor should generate its next message (EXTERNAL TRACKING)
        self.sensor_next_generation_times = {}  # Dictionary to track timing per sensor
        
        for i in range(total_sensors):
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            sensor = Sensor(id=f'mock_sensor_{i}', position=Position(x, y, z))
            
            # Track timing externally using sensor ID as key
            self.sensor_next_generation_times[sensor.id] = (
                self.current_time + random.uniform(0, self.config.data_generation_interval)
            )
            
            # Add some initial messages with realistic timing
            if random.random() < 0.6:  # 60% chance of having messages
                num_initial = random.randint(1, 5)
                for j in range(num_initial):
                    # Messages generated in the past at regular intervals
                    past_generation_time = (
                        self.current_time - 
                        ((num_initial - j) * self.config.data_generation_interval) - 
                        random.uniform(0, self.config.data_generation_interval)
                    )
                    
                    msg = DTNMessage(
                        id=f'msg_{i}_{j}',
                        source_id=sensor.id,
                        destination_id="surface_gateway",
                        data=f"mock_data_{j}",
                        generation_time=past_generation_time,
                        hop_count=0,
                        priority=1,
                        ttl=self.config.message_ttl,
                        size=self.config.message_size
                    )
                    sensor.messages.append(msg)
            
            self.mock_sensors.append(sensor)
        
        # Create realistic number of ships (more than state space)
        self.mock_ships = []
        total_ships = max(self.config.num_ships, 3)  # At least 3 for variety
        
        for i in range(total_ships):
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            
            ship = Ship(id=f'mock_ship_{i}', position=Position(x, y, 0.0))
            self.mock_ships.append(ship)

    def _update_simulation_state(self):
        """Enhanced state update to match real simulation behavior"""
        
        # Track time delta for continuous message generation
        time_delta = self.current_time - getattr(self, 'last_sim_update', 0)
        self.last_sim_update = self.current_time
        
        # Generate messages that should have occurred during time delta
        for sensor in self.mock_sensors:
            if sensor.id in self.sensor_next_generation_times:
                # Generate all messages that should have been created up to current time
                while self.sensor_next_generation_times[sensor.id] <= self.current_time:
                    # Match real simulation buffer behavior: drop oldest if full
                    if len(sensor.messages) >= self.config.sensor_buffer_capacity:
                        sensor.messages.pop(0)  # FIFO like real simulation
                    
                    gen_time = self.sensor_next_generation_times[sensor.id]
                    new_msg = sensor.generate_message(gen_time, self.config)
                    sensor.messages.append(new_msg)
                    
                    # Schedule next message generation
                    self.sensor_next_generation_times[sensor.id] += self.config.data_generation_interval
        
        # TTL cleanup (match real simulation frequency)
        if not hasattr(self, 'last_ttl_check'):
            self.last_ttl_check = 0
        
        if self.current_time - self.last_ttl_check >= self.config.message_ttl_check_interval:
            # Remove expired messages from sensors
            for sensor in self.mock_sensors:
                sensor.messages = [msg for msg in sensor.messages 
                                if (self.current_time - msg.generation_time) <= msg.ttl]
            
            # Remove expired messages from drone
            self.mock_drone.messages = [msg for msg in self.mock_drone.messages
                                    if (self.current_time - msg.generation_time) <= msg.ttl]
            
            self.last_ttl_check = self.current_time

    def close(self):
        """Clean up environment"""
        pass