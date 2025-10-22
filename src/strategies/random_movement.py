import random
from typing import List 
from .movement_strategy import MovementStrategy
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

from ..agents.sensor import Sensor
from ..agents.drone import Drone
from ..agents.ship import Ship

class RandomMovementStrategy(MovementStrategy):

    #strategy that selects a random target position within defined bounds

    def get_next_target(self, drone: Drone, sensors: List[Sensor], ships: List[Ship], other_drones: List[Drone],
                        config: SimulationConfig, current_time: float) -> Position:
   
        buffer_usage = len(drone.messages) / config.drone_buffer_capacity

        if buffer_usage >= config.random_strat_buffer_threshold:
            if ships:   
                closest_ship = self.find_closest_ship(drone, ships)
                return closest_ship.position
        
        if drone.last_visited == "ship" and buffer_usage < config.random_strat_buffer_threshold:
            if sensors:
                sensor = random.choice(sensors)
                return sensor.position
            
        if len(drone.messages) > 0 and random.random() < config.visit_ship_probability:
            if ships:
                closest_ship = self.find_closest_ship(drone, ships)
                return closest_ship.position
            
        if sensors:
            sensor = random.choice(sensors)
            return sensor.position
        
        return drone.position #fallback to current position if no other targets available
    
    def get_strategy_name(self) -> str:
        return "Random Movement Strategy"

