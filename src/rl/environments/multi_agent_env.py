"""Multi-Agent OpenAI Gym environment for DTN drone simulation"""

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


class MultiAgentDTNEnvironment(gym.Env):
    """Multi-Agent OpenAI Gym environment for DTN drone fleet simulation."""

    def __init__(self, config: SimulationConfig):
        super().__init__()

        self.config = config
        self.state_manager = RLStateManager(config)
        
        # Multi-agent configuration
        self.num_drones = config.rl_num_drones  # number of drones in the multi-agent environment
        
        # State space: Combined observations from all drones + global state
        # Each drone has same state as single agent: 7 + (3 * K) + (3 * M)
        single_drone_state_dim = 7 + (3 * config.sensors_state_space) + (3 * config.ships_state_space)
        global_state_dim = 10  # Global system metrics
        total_state_dim = (single_drone_state_dim * self.num_drones) + global_state_dim
        
        # Fixed: Use wider bounds to handle normalization edge cases
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(total_state_dim,), dtype=np.float32
        )

        # Action space: Each drone has independent action space (same as single agent)
        single_action_space = config.sensors_state_space + config.ships_state_space + 1
        self.action_space = spaces.MultiDiscrete([single_action_space] * self.num_drones)

        # Episode tracking
        self.current_step = 0
        self.max_steps = self.config.max_episode_steps
        self.episode_reward = 0.0
        self.current_time = 0.0

        # Multi-agent simulation components
        self.mock_drones: List[Drone] = []
        self.mock_sensors: List[Sensor] = []
        self.mock_ships: List[Ship] = []

        # Initialize sensor timing tracking (same as single agent)
        self.sensor_next_generation_times = {}

        # Multi-agent episode statistics
        self.episode_stats = {
            'system_messages_delivered': 0,
            'system_messages_collected': 0,
            'total_distance_traveled': 0.0,
            'system_sensor_visits': 0,
            'system_ship_visits': 0,
            'system_explore_actions': 0,
            'drone_stats': []  # Per-drone statistics
        }

        # Multi-agent AoI tracking
        self.aoi_metrics = {
            "delivered_messages": [],       
            "collected_messages": [],       
            "episode_start_time": 0.0,
            "episode_end_time": 0.0,
            "drone_interactions": []        # Track which drone did what
        }

    def reset(self, seed=None, options=None):
        """Reset environment state for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sensor_next_generation_times = {}
        self._initialize_mock_simulation()

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_time = 0.0
        
        # Reset multi-agent statistics
        self.episode_stats = {
            'system_messages_delivered': 0,
            'system_messages_collected': 0,
            'total_distance_traveled': 0.0,
            'system_sensor_visits': 0,
            'system_ship_visits': 0,
            'system_explore_actions': 0,
            'drone_stats': [
                {
                    'messages_delivered': 0,
                    'messages_collected': 0,
                    'distance_traveled': 0.0,
                    'sensor_visits': 0,
                    'ship_visits': 0,
                    'explore_actions': 0
                } for _ in range(self.num_drones)
            ]
        }
        
        # Reset AoI tracking
        self.aoi_metrics = {
            "delivered_messages": [],
            "collected_messages": [],
            "episode_start_time": self.current_time,
            "episode_end_time": 0.0,
            "drone_interactions": []
        }

        # Get initial multi-agent state
        initial_state = self._get_multi_agent_observation()

        info = {
            "episode_step": 0, 
            "simulation_time": 0.0, 
            "stats": self.episode_stats.copy(),
            "num_drones": self.num_drones
        }
        return initial_state, info

    def step(self, actions):
        """Execute actions for all drones simultaneously."""
        if not self.mock_drones:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        
        # actions is array: [drone0_action, drone1_action, drone2_action, ...]
        if len(actions) != self.num_drones:
            raise ValueError(f"Expected {self.num_drones} actions, got {len(actions)}")
        
        # Store previous positions for distance calculation
        prev_positions = [
            Position(drone.position.x, drone.position.y, drone.position.z) 
            for drone in self.mock_drones
        ]

        # Execute each drone's action simultaneously
        total_reward = 0.0
        drone_rewards = []
        
        for i, (drone, action) in enumerate(zip(self.mock_drones, actions)):
            drone_reward = self._execute_drone_action(drone, action, i, prev_positions[i])
            drone_rewards.append(drone_reward)
            total_reward += drone_reward
        
        # Add multi-agent coordination rewards/penalties
        coordination_reward = self._calculate_coordination_rewards()
        total_reward += coordination_reward

        # Add continuous AoI pressure (system-wide)
        aoi_pressure = self._calculate_system_aoi_pressure()
        total_reward += aoi_pressure

        # Update simulation state (same as single agent)
        self._update_simulation_state()

        # Get new multi-agent observation
        new_state = self._get_multi_agent_observation()

        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = self.current_time >= self.config.sim_time
        
        # Initialize info dictionary
        info = {
            "episode_step": self.current_step,
            "simulation_time": self.current_time,
            "total_reward": self.episode_reward,
            "actions_taken": actions.tolist() if hasattr(actions, 'tolist') else list(actions),
            "drone_rewards": drone_rewards,
            "coordination_reward": coordination_reward,
            "step_reward": total_reward,
            "stats": self.episode_stats.copy()
        }
        
        # Episode-end processing
        if done or truncated:
            self.aoi_metrics["episode_end_time"] = self.current_time
            
            # Calculate multi-agent AoI metrics
            multi_agent_aoi_metrics = self._calculate_multi_agent_aoi_metrics()
            
            # Calculate episode-end penalty
            episode_end_penalty = self._calculate_episode_end_aoi_penalty(multi_agent_aoi_metrics)
            total_reward += episode_end_penalty
            
            # Add comprehensive multi-agent metrics to info
            info.update({
                "multi_agent_aoi_metrics": multi_agent_aoi_metrics,
                "detailed_aoi_data": self.aoi_metrics,
                "episode_end_penalty": episode_end_penalty,
                "system_undelivered_messages": self._count_system_undelivered_messages()
            })
        
        # Update episode reward
        self.episode_reward += total_reward
        
        return new_state, total_reward, done, truncated, info

    def _get_multi_agent_observation(self):
        """Get combined observation for all drones + global system state with bounds checking."""
        drone_states = []
        
        # Get individual drone states with bounds checking
        for drone in self.mock_drones:
            drone_state = self.state_manager.get_drone_state(
                drone, self.mock_sensors, self.mock_ships, self.current_time
            )
            # Ensure drone state is within bounds
            drone_state = np.clip(drone_state, -5.0, 5.0)
            drone_states.append(drone_state)
        
        # Add global system state
        global_state = self._get_global_system_state()
        
        # Combine all states: [drone0_state, drone1_state, ..., global_state]
        combined_state = np.concatenate(drone_states + [global_state])
        
        # Final bounds check to ensure all values are within observation space
        combined_state = np.clip(combined_state, -5.0, 5.0).astype(np.float32)
        return combined_state
    
    def _get_global_system_state(self):
        """Get system-wide state information for multi-agent coordination."""
        # Count total messages in system
        total_messages_in_sensors = sum(len(sensor.messages) for sensor in self.mock_sensors)
        total_messages_in_drones = sum(len(drone.messages) for drone in self.mock_drones)
        
        # Count drones with messages
        drones_with_messages = sum(1 for drone in self.mock_drones if len(drone.messages) > 0)
        
        # Calculate system load distribution
        max_drone_load = max(len(drone.messages) for drone in self.mock_drones) if self.mock_drones else 0
        min_drone_load = min(len(drone.messages) for drone in self.mock_drones) if self.mock_drones else 0
        
        # Calculate spatial distribution (how spread out are the drones)
        drone_positions = [drone.position for drone in self.mock_drones]
        avg_drone_distance = self._calculate_average_drone_separation(drone_positions)
        
        # System urgency (average message age across all undelivered messages)
        total_age = 0.0
        total_undelivered = 0
        
        for sensor in self.mock_sensors:
            for msg in sensor.messages:
                total_age += (self.current_time - msg.generation_time)
                total_undelivered += 1
        
        for drone in self.mock_drones:
            for msg in drone.messages:
                total_age += (self.current_time - msg.generation_time)
                total_undelivered += 1
        
        system_urgency = total_age / total_undelivered if total_undelivered > 0 else 0.0
        
        # Safe normalization with clipping to prevent out-of-bounds values
        global_state = np.array([
            np.clip(total_messages_in_sensors / 100.0, -1.0, 1.0),           # Normalized sensor load
            np.clip(total_messages_in_drones / 50.0, -1.0, 1.0),             # Normalized drone load
            np.clip(drones_with_messages / max(self.num_drones, 1), 0.0, 1.0),      # Fraction of drones carrying messages
            np.clip(max_drone_load / max(self.config.drone_buffer_capacity, 1), 0.0, 1.0),  # Max drone load ratio
            np.clip(min_drone_load / max(self.config.drone_buffer_capacity, 1), 0.0, 1.0),  # Min drone load ratio
            np.clip(avg_drone_distance / 2000.0, 0.0, 1.0),                 # Normalized drone separation (increased denominator)
            np.clip(system_urgency / max(self.config.message_ttl, 1), 0.0, 1.0),    # Normalized system urgency
            np.clip(self.current_time / max(self.config.sim_time, 1), 0.0, 1.0),    # Time progress
            np.clip(len(self.mock_sensors) / 50.0, 0.0, 1.0),               # Normalized sensor count
            np.clip(len(self.mock_ships) / 10.0, 0.0, 1.0)                  # Normalized ship count
        ], dtype=np.float32)
        
        return global_state
    
    def _calculate_average_drone_separation(self, positions: List[Position]) -> float:
        """Calculate average distance between all drone pairs."""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_distance += positions[i].distance_to(positions[j])
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0

    def _execute_drone_action(self, drone: Drone, action: int, drone_idx: int, 
                             prev_position: Position) -> float:
        """Execute action for a specific drone (similar to single agent)."""
        
        # Convert action to target (same logic as single agent)
        target_position, target_entity, action_type = self._action_to_target(action, drone)
        
        # Calculate movement
        travel_distance = prev_position.distance_to(target_position)
        travel_time = travel_distance / self.config.drone_speed if self.config.drone_speed > 0 else 0.0
        
        # Move drone to target
        drone.position = target_position
        self.episode_stats['drone_stats'][drone_idx]['distance_traveled'] += travel_distance
        self.episode_stats['total_distance_traveled'] += travel_distance
        
        # Time passes for this action
        self.current_time += travel_time
        
        reward = 0.0
        
        # Sensor interaction (same logic as single agent but with drone tracking)
        if action_type == "sensor" and target_entity:
            self.episode_stats['drone_stats'][drone_idx]['sensor_visits'] += 1
            self.episode_stats['system_sensor_visits'] += 1
            
            if target_entity.has_messages():
                collected_reward = 0.0
                available_space = self.config.drone_buffer_capacity - len(drone.messages)
                messages_collected = 0
                
                while (available_space > 0 and target_entity.has_messages()):
                    message = target_entity.get_next_message_for_collection()
                    if message:
                        # AoI tracking with drone info
                        collection_aoi = self.current_time - message.generation_time
                        age_ratio = min(collection_aoi / self.config.message_ttl, 1.0)
                        
                        self.aoi_metrics["collected_messages"].append({
                            "message_id": message.id,
                            "collection_time": self.current_time,
                            "generation_time": message.generation_time,
                            "collection_aoi": collection_aoi,
                            "sensor_id": target_entity.id,
                            "drone_id": drone.id,
                            "drone_idx": drone_idx
                        })
                        
                        # Progressive collection reward (same as single agent)
                        urgency_reward = (
                            self.config.reward_collection_base + 
                            (age_ratio * self.config.reward_collection_urgency_multiplier)
                        )
                        collected_reward += urgency_reward
                        
                        drone.messages.append(message)
                        messages_collected += 1
                        available_space -= 1
                    else:
                        break
                
                if messages_collected > 0:
                    self.episode_stats['drone_stats'][drone_idx]['messages_collected'] += messages_collected
                    self.episode_stats['system_messages_collected'] += messages_collected
                    reward += collected_reward
            else:
                # Penalty for visiting empty sensor
                reward += self.config.penalty_empty_sensor
        
        # Ship interaction (same logic as single agent but with drone tracking)
        elif action_type == "ship" and target_entity:
            self.episode_stats['drone_stats'][drone_idx]['ship_visits'] += 1
            self.episode_stats['system_ship_visits'] += 1
            
            if len(drone.messages) > 0:
                delivery_reward = 0.0
                messages_delivered = len(drone.messages)
                
                # Track AoI for each delivered message
                for message in drone.messages:
                    delivery_aoi = self.current_time - message.generation_time
                    age_ratio = min(delivery_aoi / self.config.message_ttl, 1.0)
                    freshness_ratio = 1.0 - age_ratio
                    
                    self.aoi_metrics["delivered_messages"].append({
                        "message_id": message.id,
                        "delivery_time": self.current_time,
                        "generation_time": message.generation_time,
                        "delivery_aoi": delivery_aoi,
                        "ship_id": target_entity.id,
                        "drone_id": drone.id,
                        "drone_idx": drone_idx
                    })
                    
                    # Progressive delivery reward (same as single agent)
                    freshness_reward = (
                        self.config.reward_delivery_base + 
                        (freshness_ratio * self.config.reward_delivery_freshness_multiplier)
                    )
                    delivery_reward += freshness_reward
                
                # Deliver all messages
                for message in drone.messages:
                    target_entity.receive_message(message, self.current_time)
                
                drone.messages.clear()
                
                self.episode_stats['drone_stats'][drone_idx]['messages_delivered'] += messages_delivered
                self.episode_stats['system_messages_delivered'] += messages_delivered
                reward += delivery_reward
            else:
                # Penalty for visiting ship with no messages
                reward += self.config.penalty_ship_no_messages
        
        # Explore action
        elif action_type == "explore":
            self.episode_stats['drone_stats'][drone_idx]['explore_actions'] += 1
            self.episode_stats['system_explore_actions'] += 1
            reward += self.config.penalty_explore
        
        # Movement penalty (per drone) - same as single agent
        reward += travel_time * self.config.penalty_time_per_second
        
        # Carrying penalty (per message per drone) - same as single agent
        for message in drone.messages:
            message_age = self.current_time - message.generation_time
            reward += message_age * self.config.penalty_carrying_per_age_unit
        
        # Buffer management (per drone) - same as single agent
        buffer_usage = len(drone.messages) / self.config.drone_buffer_capacity
        if buffer_usage >= 0.95:
            reward += self.config.penalty_buffer_overflow
        
        return reward
    
    def _action_to_target(self, action: int, drone: Drone) -> Tuple[Position, Optional[object], str]:
        """Convert RL action to target position for specific drone (same as single agent)."""
        K = self.config.sensors_state_space
        M = self.config.ships_state_space

        if action < K:  # Choose specific sensor
            sensor_idx = action
            if self.mock_sensors:
                # Sort by distance from THIS drone (same as single agent)
                sensor_distances = [(sensor, drone.position.distance_to(sensor.position)) 
                                   for sensor in self.mock_sensors]
                sensor_distances.sort(key=lambda x: x[1])
                
                if sensor_idx < len(sensor_distances):
                    target_sensor = sensor_distances[sensor_idx][0]
                    return target_sensor.position, target_sensor, 'sensor'
                
        elif action < K + M:  # Choose specific ship
            ship_idx = action - K
            if self.mock_ships:
                # Sort by distance from THIS drone (same as single agent)
                ship_distances = [(ship, drone.position.distance_to(ship.position)) 
                                 for ship in self.mock_ships]
                ship_distances.sort(key=lambda x: x[1])
                
                if ship_idx < len(ship_distances):
                    target_ship = ship_distances[ship_idx][0]
                    return target_ship.position, target_ship, 'ship'
        
        # Explore action: random nearby position (same as single agent)
        random_x = random.uniform(0, self.config.area_size[0])
        random_y = random.uniform(0, self.config.area_size[1])
        random_z = random.uniform(self.config.min_depth, self.config.depth_range)

        return Position(random_x, random_y, random_z), None, "explore"

    def _calculate_coordination_rewards(self) -> float:
        """Calculate rewards/penalties for drone coordination."""
        coordination_reward = 0.0
        
        # 1. Penalty for multiple drones visiting the same sensor simultaneously
        current_sensor_visitors = {}
        for i, drone in enumerate(self.mock_drones):
            # Find closest sensor to each drone
            if self.mock_sensors:
                closest_sensor = min(self.mock_sensors, 
                                   key=lambda s: drone.position.distance_to(s.position))
                distance = drone.position.distance_to(closest_sensor.position)
                
                # If drone is very close to sensor (visiting it)
                if distance < 50.0:  # 50m threshold for "visiting"
                    if closest_sensor.id not in current_sensor_visitors:
                        current_sensor_visitors[closest_sensor.id] = []
                    current_sensor_visitors[closest_sensor.id].append(i)
        
        # Apply penalties for conflicts
        for sensor_id, visiting_drones in current_sensor_visitors.items():
            if len(visiting_drones) > 1:
                # Multiple drones at same sensor - inefficient
                conflict_penalty = len(visiting_drones) * -3.0  # Configurable interference penalty
                coordination_reward += conflict_penalty
        
        # 2. Reward for good load balancing
        drone_loads = [len(drone.messages) for drone in self.mock_drones]
        max_load = max(drone_loads) if drone_loads else 0
        min_load = min(drone_loads) if drone_loads else 0
        load_difference = max_load - min_load
        
        # Reward for balanced loads
        if load_difference <= 1:  # Well balanced
            balance_reward = 1.0  # Small coordination reward
            coordination_reward += balance_reward
        
        # 3. Reward for spatial coverage (drones spread out)
        avg_separation = self._calculate_average_drone_separation([d.position for d in self.mock_drones])
        target_separation = 200.0  # Target 200m separation
        
        if avg_separation >= target_separation:
            coverage_reward = 0.5  # Small coverage reward
            coordination_reward += coverage_reward
        
        return coordination_reward
    
    def _calculate_system_aoi_pressure(self) -> float:
        """System-wide continuous AoI pressure (same as single agent)."""
        total_aoi = 0.0
        message_count = 0
        
        # Calculate total system AoI for all undelivered messages
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                message_age = self.current_time - message.generation_time
                total_aoi += message_age
                message_count += 1
        
        for drone in self.mock_drones:
            for message in drone.messages:
                message_age = self.current_time - message.generation_time
                total_aoi += message_age
                message_count += 1
        
        if message_count > 0:
            # Small penalty proportional to mean system AoI
            mean_aoi = total_aoi / message_count
            return -mean_aoi * 0.001  # Small continuous pressure
        return 0.0

    def _initialize_mock_simulation(self):
        """Initialize mock simulation with multiple drones."""
        # Create multiple drones at different random positions
        self.mock_drones = []
        for i in range(self.num_drones):
            start_x = random.uniform(0, self.config.area_size[0])
            start_y = random.uniform(0, self.config.area_size[1])
            start_z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            drone = Drone(
                id=f"training_drone_{i}",
                position=Position(start_x, start_y, start_z),
                protocol=EpidemicProtocol(f"training_drone_{i}"),
                movement_strategy=None
            )
            self.mock_drones.append(drone)
        
        # Create sensors (same as single agent but scale up slightly)
        self.mock_sensors = []
        total_sensors = max(self.config.num_sensors, 15)  # Slightly more sensors for multiple drones
        
        self.sensor_next_generation_times = {}
        
        for i in range(total_sensors):
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            sensor = Sensor(id=f'mock_sensor_{i}', position=Position(x, y, z))
            self.sensor_next_generation_times[sensor.id] = (
                self.current_time + random.uniform(0, self.config.data_generation_interval)
            )
            
            # Add initial messages (same distribution as single agent)
            if random.random() < 0.6:  # 60% chance of having messages
                num_initial = random.randint(1, 5)  # Similar to single agent
                for j in range(num_initial):
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
        
        # Create ships (same as single agent)
        self.mock_ships = []
        total_ships = max(self.config.num_ships, 3)  # Slightly more ships for multiple drones
        
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
            
            # Remove expired messages from all drones
            for drone in self.mock_drones:
                drone.messages = [msg for msg in drone.messages
                                if (self.current_time - msg.generation_time) <= msg.ttl]
            
            self.last_ttl_check = self.current_time
    
    def _calculate_multi_agent_aoi_metrics(self) -> dict:
        """Calculate comprehensive multi-agent AoI metrics."""
        # Similar to single agent but with per-drone breakdown
        delivered_aois = [msg["delivery_aoi"] for msg in self.aoi_metrics["delivered_messages"]]
        
        # Per-drone delivered messages
        per_drone_delivered = [[] for _ in range(self.num_drones)]
        for msg in self.aoi_metrics["delivered_messages"]:
            drone_idx = msg["drone_idx"]
            per_drone_delivered[drone_idx].append(msg["delivery_aoi"])
        
        # Undelivered messages
        undelivered_aois = []
        
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                current_aoi = self.current_time - message.generation_time
                undelivered_aois.append({
                    "message_id": message.id,
                    "current_aoi": current_aoi,
                    "location": "sensor",
                    "entity_id": sensor.id
                })
        
        for i, drone in enumerate(self.mock_drones):
            for message in drone.messages:
                current_aoi = self.current_time - message.generation_time
                undelivered_aois.append({
                    "message_id": message.id,
                    "current_aoi": current_aoi,
                    "location": "drone",
                    "entity_id": drone.id,
                    "drone_idx": i
                })
        
        metrics = {
            "system": {
                "total_delivered": len(delivered_aois),
                "total_undelivered": len(undelivered_aois),
                "delivery_rate": len(delivered_aois) / (len(delivered_aois) + len(undelivered_aois)) if (len(delivered_aois) + len(undelivered_aois)) > 0 else 0.0,
                "mean_delivery_aoi": sum(delivered_aois) / len(delivered_aois) if delivered_aois else 0.0,
                "system_aoi": sum(delivered_aois) + sum(msg["current_aoi"] for msg in undelivered_aois)
            },
            "per_drone": [
                {
                    "delivered": len(per_drone_delivered[i]),
                    "collected": len([msg for msg in self.aoi_metrics["collected_messages"] if msg["drone_idx"] == i]),
                    "mean_delivery_aoi": sum(per_drone_delivered[i]) / len(per_drone_delivered[i]) if per_drone_delivered[i] else 0.0,
                    "drone_stats": self.episode_stats['drone_stats'][i]
                } for i in range(self.num_drones)
            ]
        }
        
        return metrics
    
    def _calculate_episode_end_aoi_penalty(self, metrics: dict) -> float:
        """Calculate episode-end penalty (same logic as single agent)."""
        total_penalty = 0.0
        
        # Progressive penalties for messages still in sensors
        for sensor in self.mock_sensors:
            for message in sensor.messages:
                message_age = self.current_time - message.generation_time
                age_penalty = (
                    self.config.penalty_undelivered_base + 
                    (message_age * self.config.penalty_undelivered_age_multiplier)
                )
                uncollected_penalty = age_penalty * self.config.penalty_uncollected_multiplier
                total_penalty += uncollected_penalty
        
        # Progressive penalties for messages in drone buffers
        for drone in self.mock_drones:
            for message in drone.messages:
                message_age = self.current_time - message.generation_time
                age_penalty = (
                    self.config.penalty_undelivered_base + 
                    (message_age * self.config.penalty_undelivered_age_multiplier)
                )
                total_penalty += age_penalty
        
        return total_penalty
    
    def _count_system_undelivered_messages(self) -> dict:
        """Count undelivered messages across system."""
        sensor_messages = sum(len(sensor.messages) for sensor in self.mock_sensors)
        drone_messages = sum(len(drone.messages) for drone in self.mock_drones)
        
        return {
            "in_sensors": sensor_messages,
            "in_drones": drone_messages,
            "total": sensor_messages + drone_messages,
            "per_drone": [len(drone.messages) for drone in self.mock_drones]
        }

    def close(self):
        """Clean up environment"""
        pass