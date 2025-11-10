#This file has all simulation configuration parameters gathered in one place for easy access and modification.
#Currently these parameters are just guesses

from dataclasses import dataclass
from typing import Tuple, List, Dict

@dataclass
class SimulationConfig:

    #DTN and Strategy choice
    dtn_protocol: str = "epidemic"  # Options: "Epidemic", "SprayAndWait", "PRoPHET"
    drone_strategy: str = "random"   # Options: "Random", "BufferAware


    #General simulation parameters
    sim_time: int = 86400  # Total simulation time in seconds (this is 24 hours)
    area_size: Tuple[float, float] = (1000.0, 1000.0)  # Size of the simulation area (width, height) in meters
    depth_range: float = 200.0  # Maximum depth in meters
    min_depth: float = 150.0  # Minimum depth in meters
    min_distance_between_sensors: float = 150.0  # Minimum distance in meters between entities at initialization
    min_distance_between_ships: float = 300.0  # Minimum distance in meters between ships at initialization
    min_distance_between_drones: float = 50.0  # Minimum distance in meters between drones at initialization
    num_sensors: int = 20  # Number of static sensor nodes
    num_drones: int = 5  # Number of mobile drone nodes
    num_ships: int = 2  # Number of surface ship nodes
    message_ttl_check_interval: float = 30.0  # Interval in seconds to check for expired messages
    statistics_interval: float = 300.0  # Interval in seconds to collect and print statistics

    # Optical communication ranges
    sensor_comm_range: float = 20.0      # 20m (short range, high speed)
    drone_comm_range: float = 50.0       # 50m (drones have better optics)
    ship_comm_range: float = 100.0       # 100m (surface has more power)

    #Sensor parameters
    data_generation_interval: float = 300.0  # Interval in seconds at which sensors generate data (5 minutes)
    sensor_buffer_capacity: int = 100  # Maximum number of messages a sensor can store

    #Drone parameters
    drone_speed: float = 2.0  # Speed of drones in m/s
    drone_buffer_capacity: int = 1000  # Maximum number of messages a drone can store
    communication_wait_time: float = 10.0  # Time in seconds a drone waits to communicate when in range
    drone_wait_no_action_time: float = 3.0  # Time in seconds a drone waits when no action is possible

    movement_time_step: float = 1.0  # Time step in seconds for drone movement updates
    encounter_communication_time: float = 5.0  # Time in seconds allocated for communication during encounters

    #Drone random strategy parameters
    visit_ship_probability: float = 0.2  # Probability of visiting a ship when deciding next target
    random_strat_buffer_threshold: float = 0.9  # Threshold of buffer usage to trigger ship visit
    
    #Ship parameters
    


    #DTN Protocol parameters


    #DTN Message parameters
    message_ttl: float = 86400.0  # Time-to-live for messages (1 day)
    message_size: int = 100  # Size of each message in bytes

DEFAULT_CONFIG = SimulationConfig()