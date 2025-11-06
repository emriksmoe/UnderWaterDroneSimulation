#This file creates all the indevidual agets used in the simulation with starting position

import random
from typing import List, Tuple

from ..agents.drone import Drone
from ..agents.sensor import Sensor
from ..agents.ship import Ship
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

#These values will be chosen later, to compare strategies and protocols
from ..protocols.dtn_protocol import EpidemicProtocol
from ..strategies.random_movement import RandomMovementStrategy

class AgentFactory:
    """Factory class for creating agents in the simulation"""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_random_position(self, entity_type: str) -> Position:
        x = random.uniform(0, self.config.area_size[0])
        y = random.uniform(0, self.config.area_size[1])

        if entity_type == "sensor":
            z = random.uniform(self.config.min_depth, self.config.depth_range) #Sensors placed at certain depth
        elif entity_type == "drone": 
            z = random.uniform(10, self.config.min_depth)
        elif entity_type == "ship":
            z = 0.0  # Ships are on the surface
        else:
            raise ValueError("Unknown entity type for position generation")
        return Position(x, y, z)

    def is_too_close(self, new_position: Position, existing_positions: List[Position], min_distance: float):
        return any(new_position.distance_to(pos) < min_distance for pos in existing_positions)

    def generate_position_with_constraints(self, entity_type: str, existing_positions: List[Position], min_distance: float, max_attempts: int = 100) -> Position:
        for attempt in range(max_attempts):
            position = self.generate_random_position(entity_type)
            if not self.is_too_close(position, existing_positions, min_distance):
                return position
        raise RuntimeError(f"Could not place {entity_type} without violating distance constraints after {max_attempts} attempts.")
    
    def create_sensors(self) -> List[Sensor]:
        sensors = []
        sensor_positions = []

        print(f"Creating {self.config.num_sensors} sensors...")

        for i in range(self.config.num_sensors):
            position = self.generate_position_with_constraints("sensor", sensor_positions, self.config.min_distance_between_sensors)
            sensor_positions.append(position)
            sensor = Sensor(id=f"sensor_{i+1}", position=position)
            sensors.append(sensor)
            print(f"Created Sensor {sensor.id} at position {sensor.position}")
        return sensors
    
    def create_drones(self) -> List[Drone]:
        drones = []
        drone_positions = []

        print(f'Creating {self.config.num_drones} drones...')

        for i in range(self.config.num_drones):
            position = self.generate_position_with_constraints("drone", drone_positions, self.config.min_distance_between_drones)
            drone_positions.append(position)

            drone = Drone(
                id = f"drone_{i+1}",
                position = position,
                protocol=EpidemicProtocol(f"drone_{i+1}"),
                movement_strategy=RandomMovementStrategy() #This is hardcoded for now
            )
            drones.append(drone)
            print(f"Created Drone {drone.id} at position {drone.position}")
        return drones
    
    def create_ships(self) -> List[Ship]:
        ships = []
        ship_positions = []

        print(f'Creating {self.config.num_ships} ships...')

        for i in range(self.config.num_ships):
            position = self.generate_position_with_constraints("ship", ship_positions, self.config.min_distance_between_ships)
            ship_positions.append(position)

            ship = Ship(
                id = f"ship_{i+1}",
                position = position
            )
            ships.append(ship)
            print(f"Created Ship {ship.id} at position {ship.position}")
        return ships
    
    def create_all_agents(self) -> Tuple[List[Sensor], List[Drone], List[Ship]]:
        sensors = self.create_sensors()
        drones = self.create_drones()
        ships = self.create_ships()
        return sensors, drones, ships

