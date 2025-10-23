from dataclasses import dataclass, field
from typing import List, Dict, Set

from ..protocols.dtn_protocol import DTNMessage
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig


@dataclass
class Ship:
    id: str
    position: Position
    received_messages: List[DTNMessage] = field(default_factory=list)
    delivery_log: Dict[str, float] = field(default_factory=dict) #message_id -> delivery_time
    seen_messages: Set[str] = field(default_factory=set)

    def receive_message(self, message: DTNMessage, current_time: float) -> bool:
        """Receive a DTN Message, we only accept it if we haven't received it before."""

        #check if we have seen this message before
        if self.has_seen_message(message.id):
            print(f"Ship {self.id} has already seen message {message.id}, not receiving again.")
            return False
        
        # Accept the message
        self.received_messages.append(message)
        self.delivery_log[message.id] = current_time
        self.seen_messages.add(message.id)

        # Calculate AoI
        aoi = current_time - message.generation_time
        print(f"Ship {self.id} received message {message.id} at time {current_time}, AoI: {aoi}")

        return True

    def has_seen_message(self, message_id: str) -> bool:
        """Check if the ship has seen a message before."""
        return message_id in self.seen_messages
    
    def can_communicate_with(self, other_position: Position, config: SimulationConfig) -> bool:
        """Check if ship can communicate with another entity"""
        return self.position.is_within_range(other_position, config.ship_comm_range)

    # Export ship data for analysis

    def get_aoi_data_for_analysis(self) -> List[Dict]:
        """Export raw AoI data for statistics analysis"""
        aoi_data = []
        for message in self.received_messages:
            if message.id in self.delivery_log:
                delivery_time = self.delivery_log[message.id]
                aoi = delivery_time - message.generation_time
                
                aoi_data.append({
                    "message_id": message.id,
                    "source_sensor": message.source_id,
                    "generation_time": message.generation_time,
                    "delivery_time": delivery_time,
                    "age_of_information": aoi,
                    "hop_count": message.hop_count,
                    "ship_id": self.id
                })
        
        return aoi_data