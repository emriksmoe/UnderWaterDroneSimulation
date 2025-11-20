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
    data_generation_interval: float = 120.0  # Interval in seconds at which sensors generate data (5 minutes)
    sensor_buffer_capacity: int = 10  # Maximum number of messages a sensor can store

    #Drone hardware parameters
    drone_speed: float = 2.0  # Speed of drones in m/s
    drone_buffer_capacity: int = 30  # Maximum number of messages a drone can store
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
    message_ttl: float = 500  # Time-to-live for messages (1 day)
    message_size: int = 100  # Size of each message in bytes
    ttl_urgency_factor: float = 0.8  # Factor to adjust urgency based on TTL remaining





    #REINFORCEMENT LEARNING CONFIGURATION PARAMETERS
    # ========================================================

    #RL state and action dimensions parameters
    sensors_state_space: int = num_sensors # Number of sensors considered in RL state representation (adds x * 3 dimensions)
    ships_state_space: int = num_ships    # Number of ships considered in RL state representation ( adds x * 3 dimensions)

    #RL drone pasrameters
    rl_num_drones: int = 5  # Number of drones in the multi drone RL environment
    
    # RL Strategy parameters
    rl_model_path: str = "models/dtn_drone_agent.pkl"
    use_pretrained_model: bool = False
    rl_training_mode: bool = False
    max_episode_steps: int = 1000


    # RL REWARD PARAMETERS - EXPONENTIAL CAMPING PUNISHMENT
    # ========================================================

    # Collection rewards - HUGE POSITIVE
    reward_collection_base: float = 1000.0  # Was 500 - DOUBLED
    reward_collection_urgency_multiplier: float = 2000.0  # Was 1000 - DOUBLED

    # Delivery rewards - MASSIVE POSITIVE
    reward_delivery_base: float = 10000.0  # Was 5000 - DOUBLED
    reward_delivery_freshness_multiplier: float = 20000.0  # Was 10000 - DOUBLED

    # Time and carrying penalties - MINIMAL
    penalty_time_per_second: float = -0.001  # Was -0.01 - REDUCED 10x
    penalty_carrying_per_age_unit: float = -0.01  # Was -0.1 - REDUCED 10x

    # Episode-end penalties - MODERATE (not overwhelming)
    penalty_undelivered_base: float = -500.0  # Was -5000 - REDUCED 10x
    penalty_undelivered_age_multiplier: float = -10.0  # Was -100 - REDUCED 10x
    penalty_uncollected_multiplier: float = 5.0  # Was 20 - REDUCED 4x

    # Action penalties - LIGHT (encourage exploration)
    penalty_empty_sensor: float = -10.0  # Was -50 - REDUCED
    penalty_ship_no_messages: float = -100.0  # Was -500 - REDUCED
    penalty_explore: float = -1.0  # Was -10 - REDUCED

    # Idle penalties - EXPONENTIALLY CATASTROPHIC (THIS IS KEY!)
    penalty_idle_at_ship: float = -10000.0  # Was -2000 - NOW 5x WORSE
    penalty_idle_at_sensor: float = -2000.0  # Was -500 - NOW 4x WORSE

    # Buffer penalties - HEAVY
    penalty_buffer_overflow: float = -10000.0  # Keep
    penalty_buffer_near_full: float = -500.0  # Keep

    # TTL expiration - CATASTROPHIC
    penalty_message_expired: float = -5000.0  # Keep
    penalty_message_expired_at_sensor: float = -10000.0  # Keep



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