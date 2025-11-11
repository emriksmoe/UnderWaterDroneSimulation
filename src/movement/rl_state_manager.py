"""State manager for reinforcement learning agents in movement tasks."""

import numpy as np
from typing import List
from src.agents.drone import Drone 
from src.agents.sensor import Sensor
from src.agents.ship import Ship
from src.config.simulation_config import SimulationConfig
from src.utils.position import Position 

class RLStateManager:
    """Manages state representation for RL agents in movement tasks."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_drone_state(self, drone: Drone, sensors: List[Sensor], ships: List[Ship], current_time: float) -> np.ndarray:
        """Converts simulation state to RL state for a drone agent, only observable information (16 dims)."""
        # 1 - Normalized drone position to [-1, 1] (3 dims)
        normalized_position = self._normalize_position(drone.position)

        # 2 - Own buffer ussage (1 dim)
        buffer_usage = len(drone.messages) / self.config.drone_buffer_capacity

        # 3 - Message AoI information (3 dims)
        message_aoi_info = self._get_message_aoi_info(drone, current_time)

        #4 - Top K nearest sensors info (K * 3 dims)
        sensor_info = self._get_top_k_sensor_info(drone, sensors, self.config.num_sensors_for_rl)

        #5 - Top K nearest ships info (M * 3 dims)
        ship_info = self._get_top_k_ship_info(drone, ships, self.config.num_ships_for_rl)

        #totalt dimentions: 3 + 1 + 3 + (K*3) + (M*3) = 
        state = np.concatenate([
            normalized_position,
            np.array([buffer_usage]),
            message_aoi_info,
            sensor_info,
            ship_info
        ])

        return state.astype(np.float32)

    def _normalize_position(self, position: Position) -> np.ndarray:
        x, y, z = position.x, position.y, position.z

        # Normalize x, y to [-1, 1] based on area size and depth
        x_norm = (x - self.config.area_size[0] / 2) / (self.config.area_size[0] / 2)
        y_norm = (y - self.config.area_size[1] / 2) / (self.config.area_size[1] / 2)
        z_norm = (z - self.config.depth_range / 2) / (self.config.depth_range / 2)

        #clamp to [-1, 1]
        x_norm = max(-1, min(1, x_norm))
        y_norm = max(-1, min(1, y_norm))
        z_norm = max(-1, min(1, z_norm))

        return np.array([x_norm, y_norm, z_norm])
    
    def _get_message_aoi_info(self, drone: Drone, current_time: float) -> np.ndarray:
        """Get AoI information about messages drone is carrying (3 dimensions)"""
        if not drone.messages:
            return np.array([0.0, 0.0, 0.0])  # No messages
        
        # Calculate AoI for all messages
        message_ages = [(current_time - msg.generation_time) for msg in drone.messages]
        
        # Average AoI (normalized to message TTL)
        avg_aoi = np.mean(message_ages) / self.config.message_ttl
        avg_aoi_norm = min(avg_aoi, 1.0)  # Clamp to [0, 1]
        
        # Maximum AoI (most urgent message)  
        max_aoi = max(message_ages) / self.config.message_ttl
        max_aoi_norm = min(max_aoi, 1.0)
        
        # Urgency level (fraction of messages approaching TTL)
        urgent_threshold = self.config.ttl_urgency_factor * self.config.message_ttl 
        urgent_count = sum(1 for age in message_ages if age > urgent_threshold)
        urgency_ratio = urgent_count / len(message_ages)
        
        return np.array([avg_aoi_norm, max_aoi_norm, urgency_ratio])

    
    def _get_top_k_sensor_info(self, drone: Drone, sensors: List[Sensor], k: int) -> np.ndarray:
        """Get info for K nearest sensors (k * 3 dimensions)"""
        if not sensors:
            return np.ones(k * 3)  # Max distance padding
        
        # Sort sensors by distance
        sensor_distances = [(sensor, drone.position.distance_to(sensor.position)) 
                        for sensor in sensors]
        sensor_distances.sort(key=lambda x: x[1])
        
        sensor_info = []
        for i in range(k):
            if i < len(sensor_distances):
                sensor, distance = sensor_distances[i]
                
                # Distance (normalized)
                distance_norm = min(distance / 1000.0, 1.0)
                
                # Direction vector
                dx = sensor.position.x - drone.position.x  
                dy = sensor.position.y - drone.position.y
                direction_magnitude = np.sqrt(dx*dx + dy*dy)
                
                if direction_magnitude > 0:
                    dx_norm = dx / direction_magnitude * 0.5  # [-0.5, 0.5]
                    dy_norm = dy / direction_magnitude * 0.5
                else:
                    dx_norm = dy_norm = 0.0
                    
                sensor_info.extend([distance_norm, dx_norm, dy_norm])
            else:
                # Padding for missing sensors
                sensor_info.extend([1.0, 0.0, 0.0])
        
        return np.array(sensor_info)
    
    def _get_top_k_ship_info(self, drone: Drone, ships: List[Ship], k: int) -> np.ndarray:
        """Get info for K nearest ships (k * 3 dimensions)"""
        if not ships:
            return np.ones(k * 3)  # Max distance padding
        
        # Sort ships by distance
        ship_distances = [(ship, drone.position.distance_to(ship.position)) 
                         for ship in ships]
        ship_distances.sort(key=lambda x: x[1])
        
        ship_info = []
        for i in range(k):
            if i < len(ship_distances):
                ship, distance = ship_distances[i]
                
                # Distance (normalized)
                distance_norm = min(distance / 1000.0, 1.0)
                
                # Direction vector
                dx = ship.position.x - drone.position.x  
                dy = ship.position.y - drone.position.y
                direction_magnitude = np.sqrt(dx*dx + dy*dy)
                
                if direction_magnitude > 0:
                    dx_norm = dx / direction_magnitude * 0.5  # [-0.5, 0.5]
                    dy_norm = dy / direction_magnitude * 0.5
                else:
                    dx_norm = dy_norm = 0.0
                    
                ship_info.extend([distance_norm, dx_norm, dy_norm])
            else:
                # Padding for missing ships
                ship_info.extend([1.0, 0.0, 0.0])
        
        return np.array(ship_info)