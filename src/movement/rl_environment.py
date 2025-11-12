"""Single agent OpenAI Gym environment for DTN drone simulation"""

import gymnasium as gym
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from gymnasium import spaces
import random

from src.config.simulation_config import SimulationConfig
from src.movement.rl_state_manager import RLStateManager
from src.agents.drone import Drone
from src.agents.sensor import Sensor
from src.agents.ship import Ship
from src.utils.position import Position
from src.protocols.dtn_protocol import DTNMessage


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

        # Episode statistics
        self.episode_stats = {
            'messages_delivered': 0,
            'messages_collected': 0,
            'total_distance_traveled': 0.0,
            'sensor_visits': 0,
            'ship_visits': 0,
            'explore_actions': 0
        }

    def reset(self, seed=None, options=None):
        """Reset environment state for a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

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

        # Execute action and get reward
        step_reward = self._execute_action(target_position, target_entity, action_type, prev_position)

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
        
        # Update episode reward
        self.episode_reward += step_reward
        
        info = {
            "episode_step": self.current_step,
            "simulation_time": self.current_time,
            "total_reward": self.episode_reward,
            "action_taken": action,
            "target_type": action_type,
            "step_reward": step_reward,
            "stats": self.episode_stats.copy()
        }
        
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
        
        # Move drone to target
        self.mock_drone.position = target_position
        self.episode_stats['total_distance_traveled'] += travel_distance
        
        reward = 0.0
        
        # Sensor interaction
        if action_type == "sensor" and target_entity:
            self.episode_stats['sensor_visits'] += 1
            
            if target_entity.has_messages():
                # Collect messages
                available_space = self.config.drone_buffer_capacity - len(self.mock_drone.messages)
                messages_collected = 0
                
                while (available_space > 0 and target_entity.has_messages()):
                    message = target_entity.get_next_message_for_collection()
                    if message:
                        self.mock_drone.messages.append(message)
                        messages_collected += 1
                        available_space -= 1
                    else:
                        break
                
                if messages_collected > 0:
                    self.episode_stats['messages_collected'] += messages_collected
                    reward += messages_collected * self.config.reward_collection
            else:
                # Penalty for visiting empty sensor
                reward += self.config.reward_idle * 0.5
        
        # Ship interaction  
        elif action_type == "ship" and target_entity and len(self.mock_drone.messages) > 0:
            messages_delivered = len(self.mock_drone.messages)
            
            # Deliver all messages
            for message in self.mock_drone.messages:
                target_entity.receive_message(message, self.current_time)
            
            self.mock_drone.messages.clear()
            
            self.episode_stats['messages_delivered'] += messages_delivered
            self.episode_stats['ship_visits'] += 1
            reward += messages_delivered * self.config.reward_delivery
        
        # Explore action
        elif action_type == "explore":
            self.episode_stats['explore_actions'] += 1
            reward += self.config.reward_idle * 0.2
        
        # Movement penalty
        reward += travel_distance * self.config.reward_movement_penalty
        
        # Buffer management
        buffer_usage = len(self.mock_drone.messages) / self.config.drone_buffer_capacity
        if buffer_usage >= 0.95:
            reward += self.config.reward_buffer_overflow
        
        return reward
    
    def _initialize_mock_simulation(self):
        """Initialize mock simulation components with realistic entity counts"""
        # Create mock drone at random position
        start_x = random.uniform(0, self.config.area_size[0])
        start_y = random.uniform(0, self.config.area_size[1])
        start_z = random.uniform(self.config.min_depth, self.config.depth_range)
        
        self.mock_drone = Drone(
            id="training_drone",
            position=Position(start_x, start_y, start_z)
        )
        
        # Create realistic number of sensors (more than state space)
        self.mock_sensors = []
        total_sensors = max(self.config.num_sensors, 8)  # At least 8 for variety
        
        for i in range(total_sensors):
            # Random placement across simulation area
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            sensor = Sensor(id=f'mock_sensor_{i}', position=Position(x, y, z))
            
            # Add some random initial messages
            if random.random() < 0.6:  # 60% chance of having messages
                for j in range(random.randint(1, 5)):
                    msg = DTNMessage(
                        id=f'msg_{i}_{j}',
                        source_id=sensor.id,
                        destination_id="surface_gateway",
                        data=f"mock_data_{j}",
                        generation_time=self.current_time - random.uniform(0, 1800),
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
        """Update simulation state"""
        time_step = 10.0
        self.current_time += time_step
        
        # Generate new messages
        for sensor in self.mock_sensors:
            if len(sensor.messages) < self.config.sensor_buffer_capacity:
                if random.random() < 0.1:
                    new_msg = sensor.generate_message(self.current_time, self.config)
                    sensor.add_message_to_buffer(new_msg, self.config)
        
        # Age out expired messages
        for sensor in self.mock_sensors:
            sensor.messages = [msg for msg in sensor.messages 
                             if (self.current_time - msg.generation_time) < msg.ttl]
        
        self.mock_drone.messages = [msg for msg in self.mock_drone.messages
                                  if (self.current_time - msg.generation_time) < msg.ttl]

    def close(self):
        """Clean up environment"""
        pass