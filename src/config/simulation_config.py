#This file has all simulation configuration parameters gathered in one place for easy access and modification.

from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class SimulationConfig:

    #General simulation parameters
    sim_time: int = 86400  # Total simulation time in seconds (this is 24 hours)
    area_size: Tuple[float, float] = (1000.0, 1000.0)  # Size of the simulation area (width, height) in meters
    depth_range: float = 200.0  # Maximum depth in meters
    num_sensors: int = 20  # Number of static sensor nodes
    num_drones: int = 5  # Number of mobile drone nodes

    #Sensor parameters
    data_generation_interval: int = 60  # Interval in seconds at which sensors generate data (1 minute)

    #Drone parameters

    
    #Ship parameters


    #DTN Protocol parameters


DEFAULT_CONFIG = SimulationConfig()