"""Multi-Agent OpenAI Gym environment for DTN drone simulation"""

import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional
from gymnasium import spaces
import random

from src.config.simulation_config import SimulationConfig
from src.rl.state_manager import RLStateManager
from src.agents.drone import Drone
from src.agents.sensor import Sensor
from src.agents.ship import Ship
from src.utils.position import Position
from src.protocols.dtn_protocol import DTNMessage
from src.protocols.dtn_protocol import EpidemicProtocol


class IndependentMultiAgentDTNEnvironment(gym.Env):
    """Multi-Agent OpenAI Gym environment for DTN drone fleet simulation."""

    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.num_drones = config.rl_num_drones