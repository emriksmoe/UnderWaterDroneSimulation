"""
Simulation Configuration
========================
Configuration parameters for the DTN simulation and RL training.
BALANCED VERSION - Good AoI optimization without excessive training time.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

@dataclass
class SimulationConfig:

    # ============================================================================
    # ACTUAL SIMULATION CONFIGURATION PARAMETERS
    # ============================================================================


    # General simulation parameters
    sim_time: int = 86400  # 24 hours (reduced from 48h for faster RL training iteration)
    area_size: Tuple[float, float] = (1000.0, 1000.0)
    depth_range: float = 200.0
    min_depth: float = 150.0
    min_distance_between_sensors: float = 200.0  # Increased from 150.0 (250m too dense for 20 sensors)
    num_sensors: int = 20  # ✅ Your original value
    num_drones: int = 1
    num_ships: int = 1
    statistics_interval: float = 300.0

    # Optical communication ranges
    sensor_comm_range: float = 20.0
    drone_comm_range: float = 50.0
    ship_comm_range: float = 100.0

    # Sensor parameters
    data_generation_interval: float = 120.0  # Default mean interval (Poisson process)
    sensor_buffer_capacity: int = 250
    
    # ✅ Sensor-specific generation rates (mean intervals in seconds)
    # Distribution: 5 fast, 10 medium, 5 slow sensors
    # This gives RL an advantage - Round-Robin treats all equally!
    # Drone hardware parameters
    drone_speed: float = 2.0
    drone_buffer_capacity: int = 800  # Reduced from 500 - forces prioritization decisions
    communication_hand_shake_time: float = 2.0
    communication_bitrate: float = 0.03


    # Drone random strategy parameters
    strategy_buffer_threshold: float = 1.0  # Return to ship only when
    
    # Message parameters
    message_size: int = 100

    # ============================================================================
    # REINFORCEMENT LEARNING CONFIGURATION PARAMETERS
    # ============================================================================



DEFAULT_CONFIG = SimulationConfig()