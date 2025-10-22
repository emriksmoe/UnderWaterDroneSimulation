#This is where the drone agent is defined

from dataclasses import dataclass, field
from typing import List, Optional
from ..protocols.dtn_protocol import DTNMessage, DTNProtocol
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

@dataclass
class Drone:
    id: str
    position: Position
    protocol: DTNProtocol
    messages: List[DTNMessage] = field(default_factory=list)
    battery_level: float = 100.0  # Battery level percentage (might be used in future extensions)
    target_position: Optional[Position] = None  # For movement logic
    last_visited: str

    def calculate_travel_time(self, target: Position, config: SimulationConfig) -> float:
        distance = self.position.distance_to(target)
        return distance / config.drone_speed  # Time = Distance / Speed
    
    
