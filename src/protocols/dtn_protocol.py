#Here the DTN protocol is implemented

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from ..config.simulation_config import SimulationConfig
import time

@dataclass 
class DTNMessage:
    id: str
    source_id: int
    data: str
    generation_time: float
    size: int # Size in bytes

    def get_AoI(self, current_time: float) -> float:
        return current_time - self.generation_time
    