from abc import ABC, abstractmethod
from typing import List, Optional
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

from ..agents.sensor import Sensor
from ..agents.drone import Drone
from ..agents.ship import Ship

class MovementStrategy(ABC):
    """Abstract base class for drone movement strategies.
    
    Defines the interface that all movement strategies must implement
    to determine where a drone should move next.
    """

    @abstractmethod 
    def get_next_target(self, drone: Drone, sensors: List[Sensor], ships: List[Ship], other_drones: List[Drone], config: SimulationConfig, current_time: float) -> Position:
        """Select the next target position for the drone""" 
        pass

    @abstractmethod #returns the name of the strategy
    def get_strategy_name(self) -> str:
        """Return the name of the strategy""" 
        pass

    # Shared utility methods

    def find_closest_ship(self, drone: Drone, ships: List[Ship]) -> Ship:
        """Find the closest ship to the drone's current position."""
        if not ships:
            raise ValueError("No ships available to find the closest one.")
        
        closest_ship = min(ships, key=lambda ship: drone.position.distance_to(ship.position))
        return closest_ship
    
    def find_closest_sensor(self, drone: Drone, sensors: List[Sensor]) -> Sensor:
        """Find the closest sensor to the drone's current position."""
        if not sensors:
            raise ValueError("No sensors available to find the closest one.")
        
        closest_sensor = min(sensors, key=lambda sensor: drone.position.distance_to(sensor.position))
        return closest_sensor