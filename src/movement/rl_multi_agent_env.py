"""Multi-Agent OpenAI Gym environment for DTN drone simulation"""

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


class MultiAgentDTNEnvironment(gym.Env):
    """Multi-Agent OpenAI Gym environment for DTN drone simulation."""

    def __init__(self, config: SimulationConfig, num_training_drones: int = 5):
        super().__init__()

        self.config = config
        self.num_training_drones = num_training_drones
        self.state_manager = RLStateManager(config)

        # Single agent spaces
        K = self.config.sensors_state_space
        M = self.config.ships_state_space
        state_dims = 7 + (3 * K) + (3 * M)
        
        single_obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_dims,), dtype=np.float32
        )
        single_action_space = spaces.Discrete(K + M + 1)

        # Multi-agent spaces - each agent has same observation/action space
        self.observation_space = spaces.Dict({
            f'agent_{i}': single_obs_space for i in range(num_training_drones)
        })
        self.action_space = spaces.Dict({
            f'agent_{i}': single_action_space for i in range(num_training_drones)
        })

        # Episode tracking
        self.current_step = 0
        self.max_steps = self.config.max_episode_steps
        self.current_time = 0.0

        # Multi-agent simulation components
        self.mock_drones: List[Drone] = []
        self.mock_sensors: List[Sensor] = []
        self.mock_ships: List[Ship] = []

        # Episode statistics (per agent)
        self.episode_stats = {}

    def reset(self, seed=None, options=None):
        """Reset environment state for a new multi-agent episode."""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._initialize_mock_simulation()

        # Reset episode tracking
        self.current_step = 0
        self.current_time = 0.0
        
        # Initialize stats for each agent
        self.episode_stats = {
            f'agent_{i}': {
                'messages_delivered': 0,
                'messages_collected': 0,
                'total_distance_traveled': 0.0,
                'sensor_visits': 0,
                'ship_visits': 0,
                'explore_actions': 0,
                'episode_reward': 0.0
            } for i in range(self.num_training_drones)
        }

        # Get initial states for all agents
        observations = {}
        for i, drone in enumerate(self.mock_drones):
            observations[f'agent_{i}'] = self.state_manager.get_drone_state(
                drone, self.mock_sensors, self.mock_ships, self.current_time
            )

        info = {
            "episode_step": 0, 
            "simulation_time": 0.0, 
            "stats": self.episode_stats.copy(),
            "num_agents": self.num_training_drones
        }
        return observations, info

    def step(self, actions):
        """Execute actions for all agents simultaneously."""
        if not self.mock_drones:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        observations = {}
        rewards = {}
        dones = {}
        truncated = {}
        infos = {}

        # Store previous positions for all drones
        prev_positions = []
        for drone in self.mock_drones:
            prev_positions.append(Position(drone.position.x, drone.position.y, drone.position.z))

        # Execute all actions simultaneously (this is where competition happens!)
        for i, drone in enumerate(self.mock_drones):
            agent_id = f'agent_{i}'
            action = actions[agent_id]
            
            # Convert action to target for this specific drone
            target_position, target_entity, action_type = self._action_to_target(action, drone)
            
            # Execute action and get reward
            step_reward = self._execute_action(
                drone, target_position, target_entity, action_type, prev_positions[i], agent_id
            )
            
            rewards[agent_id] = step_reward
            self.episode_stats[agent_id]['episode_reward'] += step_reward
            
            # Episode termination (same for all agents)
            dones[agent_id] = self.current_step >= self.max_steps
            truncated[agent_id] = self.current_time >= self.config.sim_time
            
            infos[agent_id] = {
                "action_taken": action,
                "target_type": action_type,
                "step_reward": step_reward,
                "buffer_usage": len(drone.messages) / self.config.drone_buffer_capacity,
                "messages_in_buffer": len(drone.messages)
            }

        # Update shared simulation state (affects all agents)
        self._update_simulation_state()

        # Get new observations for all agents
        for i, drone in enumerate(self.mock_drones):
            observations[f'agent_{i}'] = self.state_manager.get_drone_state(
                drone, self.mock_sensors, self.mock_ships, self.current_time
            )

        self.current_step += 1
        
        # Add global information
        all_infos = infos.copy()
        all_infos["_global_"] = {
            "episode_step": self.current_step,
            "simulation_time": self.current_time,
            "stats": self.episode_stats.copy(),
            "total_messages_in_system": sum(len(s.messages) for s in self.mock_sensors),
            "total_messages_in_drone_buffers": sum(len(d.messages) for d in self.mock_drones)
        }
        
        return observations, rewards, dones, truncated, all_infos

    def _action_to_target(self, action: int, drone: Drone) -> Tuple[Position, Optional[object], str]:
        """Convert RL action to target position for specific drone."""
        K = self.config.sensors_state_space
        M = self.config.ships_state_space

        if action < K:  # Choose specific sensor
            sensor_idx = action

            if self.mock_sensors:
                # Sort sensors by distance FROM THIS SPECIFIC DRONE
                sensor_distances = [(sensor, drone.position.distance_to(sensor.position)) 
                                   for sensor in self.mock_sensors]
                sensor_distances.sort(key=lambda x: x[1])
                
                if sensor_idx < len(sensor_distances):
                    target_sensor = sensor_distances[sensor_idx][0]
                    return target_sensor.position, target_sensor, 'sensor'
                
        elif action < K + M:  # Choose specific ship
            ship_idx = action - K

            if self.mock_ships:
                # Sort ships by distance FROM THIS SPECIFIC DRONE
                ship_distances = [(ship, drone.position.distance_to(ship.position)) 
                                 for ship in self.mock_ships]
                ship_distances.sort(key=lambda x: x[1])
                
                if ship_idx < len(ship_distances):
                    target_ship = ship_distances[ship_idx][0]
                    return target_ship.position, target_ship, 'ship'
                
        # Explore action: random position
        random_x = random.uniform(0, self.config.area_size[0])
        random_y = random.uniform(0, self.config.area_size[1])
        random_z = random.uniform(self.config.min_depth, self.config.depth_range)

        return Position(random_x, random_y, random_z), None, "explore"

    def _execute_action(self, drone: Drone, target_position: Position, target_entity: Optional[object], 
                       action_type: str, prev_position: Position, agent_id: str) -> float:
        """Execute action for specific drone and return reward."""
        
        travel_distance = prev_position.distance_to(target_position)
        
        # Move drone to target
        drone.position = target_position
        self.episode_stats[agent_id]['total_distance_traveled'] += travel_distance
        
        reward = 0.0
        
        # Sensor interaction (with multi-agent competition!)
        if action_type == "sensor" and target_entity:
            self.episode_stats[agent_id]['sensor_visits'] += 1
            
            if target_entity.has_messages():
                # Collect messages (competition: first drone to arrive gets messages!)
                available_space = self.config.drone_buffer_capacity - len(drone.messages)
                messages_collected = 0
                
                while (available_space > 0 and target_entity.has_messages()):
                    message = target_entity.get_next_message_for_collection()
                    if message:
                        drone.messages.append(message)
                        messages_collected += 1
                        available_space -= 1
                    else:
                        break
                
                if messages_collected > 0:
                    self.episode_stats[agent_id]['messages_collected'] += messages_collected
                    reward += messages_collected * self.config.reward_collection
                else:
                    # Penalty for visiting sensor with no available messages (competition effect!)
                    reward += self.config.reward_idle * 0.3
            else:
                # Penalty for visiting empty sensor
                reward += self.config.reward_idle * 0.5
        
        # Ship interaction
        elif action_type == "ship" and target_entity and len(drone.messages) > 0:
            messages_delivered = len(drone.messages)
            
            # Deliver all messages
            for message in drone.messages:
                target_entity.receive_message(message, self.current_time)
            
            drone.messages.clear()
            
            self.episode_stats[agent_id]['messages_delivered'] += messages_delivered
            self.episode_stats[agent_id]['ship_visits'] += 1
            reward += messages_delivered * self.config.reward_delivery
        
        # Explore action
        elif action_type == "explore":
            self.episode_stats[agent_id]['explore_actions'] += 1
            reward += self.config.reward_idle * 0.2
        
        # Movement penalty (encourages efficient movement)
        reward += travel_distance * self.config.reward_movement_penalty
        
        # Buffer management penalty
        buffer_usage = len(drone.messages) / self.config.drone_buffer_capacity
        if buffer_usage >= 0.95:
            reward += self.config.reward_buffer_overflow
        
        # Cooperation bonus: reward for maintaining diverse spatial distribution
        cooperation_bonus = self._calculate_cooperation_bonus(drone, agent_id)
        reward += cooperation_bonus
        
        return reward

    def _calculate_cooperation_bonus(self, drone: Drone, agent_id: str) -> float:
        """Calculate bonus reward for spatial cooperation (avoiding clustering)"""
        cooperation_bonus = 0.0
        
        # Check distances to other drones
        min_distance_to_other = float('inf')
        for other_drone in self.mock_drones:
            if other_drone.id != drone.id:
                distance = drone.position.distance_to(other_drone.position)
                min_distance_to_other = min(min_distance_to_other, distance)
        
        # Reward for maintaining good spacing (discourage clustering)
        area_diagonal = np.sqrt(self.config.area_size[0]**2 + self.config.area_size[1]**2)
        good_spacing_threshold = area_diagonal * 0.15  # 15% of diagonal
        too_close_threshold = area_diagonal * 0.08     # 8% of diagonal
        
        if min_distance_to_other > good_spacing_threshold:
            cooperation_bonus += 0.1
        elif min_distance_to_other < too_close_threshold:
            cooperation_bonus -= 0.1
        
        return cooperation_bonus

    def _initialize_mock_simulation(self):
        """Initialize mock simulation with multiple competing drones."""
        
        # Create multiple real drone objects
        self.mock_drones = []
        for i in range(self.num_training_drones):
            start_x = random.uniform(0, self.config.area_size[0])
            start_y = random.uniform(0, self.config.area_size[1])
            start_z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            drone = Drone(
                id=f"training_drone_{i}",
                position=Position(start_x, start_y, start_z)
            )
            self.mock_drones.append(drone)

        # Create realistic number of sensors (shared resource for competition)
        self.mock_sensors = []
        total_sensors = max(self.config.num_sensors, self.num_training_drones * 2)  # At least 2 sensors per drone
        
        for i in range(total_sensors):
            # Spread sensors across simulation area
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            z = random.uniform(self.config.min_depth, self.config.depth_range)
            
            sensor = Sensor(id=f'mock_sensor_{i}', position=Position(x, y, z))
            
            # Add initial messages (competition resource)
            if random.random() < 0.7:  # 70% chance of having messages
                for j in range(random.randint(1, 8)):  # More messages for competition
                    msg = DTNMessage(
                        id=f'msg_{i}_{j}',
                        source_id=sensor.id,
                        destination_id="surface_gateway",
                        data=f"mock_data_{j}",
                        generation_time=self.current_time - random.uniform(0, 1800),
                        hop_count=0,
                        priority=random.randint(1, 3),  # Varied priorities
                        ttl=self.config.message_ttl,
                        size=self.config.message_size
                    )
                    sensor.messages.append(msg)
            
            self.mock_sensors.append(sensor)

        # Create ships (delivery points)
        self.mock_ships = []
        total_ships = max(self.config.num_ships, 3)  # At least 3 ships for variety
        
        for i in range(total_ships):
            x = random.uniform(0, self.config.area_size[0])
            y = random.uniform(0, self.config.area_size[1])
            
            ship = Ship(id=f'mock_ship_{i}', position=Position(x, y, 0.0))
            self.mock_ships.append(ship)

    def _update_simulation_state(self):
        """Update shared simulation state (affects all agents)."""
        time_step = 10.0
        self.current_time += time_step
        
        # Generate new messages at sensors (continuous resource replenishment)
        for sensor in self.mock_sensors:
            if len(sensor.messages) < self.config.sensor_buffer_capacity:
                # Higher message generation rate for multi-agent competition
                if random.random() < 0.15:  # 15% chance per time step
                    new_msg = sensor.generate_message(self.current_time, self.config)
                    sensor.add_message_to_buffer(new_msg, self.config)
        
        # Age out expired messages from sensors
        for sensor in self.mock_sensors:
            sensor.messages = [msg for msg in sensor.messages 
                             if (self.current_time - msg.generation_time) < msg.ttl]
        
        # Age out expired messages from all drone buffers
        for drone in self.mock_drones:
            drone.messages = [msg for msg in drone.messages
                             if (self.current_time - msg.generation_time) < msg.ttl]

    def get_agent_ids(self):
        """Return list of agent IDs for multi-agent training frameworks."""
        return [f'agent_{i}' for i in range(self.num_training_drones)]

    def get_env_info(self):
        """Return environment information for multi-agent training."""
        return {
            "num_agents": self.num_training_drones,
            "observation_space": self.observation_space[f'agent_0'],  # Same for all agents
            "action_space": self.action_space[f'agent_0'],  # Same for all agents
            "state_shape": self.observation_space[f'agent_0'].shape,
            "action_shape": self.action_space[f'agent_0'].n
        }

    def close(self):
        """Clean up environment"""
        pass
