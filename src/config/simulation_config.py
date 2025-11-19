#This file has all simulation configuration parameters gathered in one place for easy access and modification.
#Currently these parameters are just guesses

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

@dataclass
class SimulationConfig:

    # ACTUAL SIMULATION CONFIGURATION PARAMETERS
    # ========================================================

    #DTN and Strategy choice
    dtn_protocol: str = "epidemic"  # Options: "Epidemic", "SprayAndWait", "PRoPHET"
    drone_strategy: str = "random"   # Options: "Random", "BufferAware

    #General simulation parameters
    sim_time: int = 86400  # Total simulation time in seconds (this is 24 hours)
    movement_time_step: float = 1.0  # Time step in seconds for drone movement updates
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

    #Drone hardware parameters
    drone_speed: float = 2.0  # Speed of drones in m/s
    drone_buffer_capacity: int = 1000  # Maximum number of messages a drone can store
    communication_wait_time: float = 10.0  # Time in seconds a drone waits to communicate when in range
    drone_wait_no_action_time: float = 3.0  # Time in seconds a drone waits when no action is possible
    encounter_communication_time: float = 5.0  # Time in seconds allocated for communication during encounters

    #Drone random strategy parameters
    visit_ship_probability: float = 0.2  # Probability of visiting a ship when deciding next target
    random_strat_buffer_threshold: float = 0.9  # Threshold of buffer usage to trigger ship visit
    encounter_cooldown_time: float = 60.0  # Minimum time between encounters with same drone (seconds)
    
    #Ship parameters
    
    #DTN Protocol parameters

    #DTN Message parameters
    message_ttl: float = 86400.0  # Time-to-live for messages (1 day)
    message_size: int = 100  # Size of each message in bytes
    ttl_urgency_factor: float = 0.8  # Factor to adjust urgency based on TTL remaining





    #REINFORCEMENT LEARNING CONFIGURATION PARAMETERS
    # ========================================================

    #RL state and action dimensions parameters
    sensors_state_space: int = num_sensors # Number of sensors considered in RL state representation (adds x * 3 dimensions)
    ships_state_space: int = num_ships    # Number of ships considered in RL state representation ( adds x * 3 dimensions)

    #RL drone pasrameters
    rl_num_drones: int = num_drones  # Number of drones in the RL environment
    
    # RL Strategy parameters
    rl_model_path: str = "models/dtn_drone_agent.pkl"
    use_pretrained_model: bool = False
    rl_training_mode: bool = False
    max_episode_steps: int = 1000


    # RL reward parameters  
    reward_collection_base: float = 5.0              # Base reward for any collection
    reward_collection_urgency_multiplier: float = 15.0  # Multiplier for urgency (age_ratio)
    
    # Delivery rewards - scale with freshness (inverse of AoI)
    reward_delivery_base: float = 10.0               # Base reward for any delivery
    reward_delivery_freshness_multiplier: float = 25.0  # Multiplier for freshness (1 - age_ratio)
    
    # Time and carrying penalties
    penalty_time_per_second: float = -0.1             # Penalty per second of travel
    penalty_carrying_per_age_unit: float = -0.01      # Penalty per second of message age carried
    
    # Episode-end progressive penalties
    penalty_undelivered_base: float = -10.0          # Base penalty for any undelivered message
    penalty_undelivered_age_multiplier: float = -0.05 # Penalty multiplier per second of age
    penalty_uncollected_multiplier: float = 2.0      # Extra multiplier for never-collected messages
    
    # Action penalties
    penalty_empty_sensor: float = -2.0
    penalty_ship_no_messages: float = -1.0
    penalty_explore: float = -0.5
    penalty_buffer_overflow: float = -10.0
  

    # RL environment parameters   # Number of discrete actions




    # REINFORCEMENT LEARNING CONFIGURATION TRAINING HYPERPARAMETERS
    # ========================================================

    # Network Architecture
    dqn_net_arch: List[int] = field(default_factory=lambda: [256, 256, 128])  # Neural network architecture
    dqn_activation: str = "ReLU"    # Activation function
    
    # Learning Parameters  
    dqn_learning_rate: float = 1e-4
    dqn_gamma: float = 0.95         # Discount factor (lower for AoI urgency)
    
    # Experience Replay
    dqn_buffer_size: int = 50000
    dqn_learning_starts: int = 1000
    dqn_batch_size: int = 64
    
    # Target Network
    dqn_target_update_interval: int = 1000
    dqn_tau: float = 1.0           # Hard target updates
    
    # Exploration Strategy
    dqn_exploration_fraction: float = 0.3
    dqn_exploration_initial_eps: float = 1.0
    dqn_exploration_final_eps: float = 0.05
    
    # Training Configuration
    dqn_total_timesteps: int = 100000
    dqn_train_freq: int = 4
    dqn_gradient_steps: int = 1
    
    # Training Pipeline
    dqn_eval_freq: int = 2000
    dqn_n_eval_episodes: int = 5
    dqn_log_interval: int = 100
    dqn_verbose: int = 1
    dqn_seed: int = 42
    
    # Logging and Saving
    dqn_tensorboard_log: str = "./logs"
    dqn_tb_log_name: str = "dqn_aoi_training"
    dqn_save_final_model: bool = True

    #Testing
    test_episodes: int = 3



DEFAULT_CONFIG = SimulationConfig()