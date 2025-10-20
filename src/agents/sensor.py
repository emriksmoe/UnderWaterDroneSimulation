#This is where the sensor agent is defined

from dataclasses import dataclass, field
from typing import List, Tuple
import simpy
from ..protocols.dtn_protocol import DTNMessage
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

@dataclass
class Sensor:
    id: str
    position: Position
    messages: List[DTNMessage] = field(default_factory=list)
    

    @property
    def data_generation_interval(self) -> int:
        return self.config.data_generation_interval