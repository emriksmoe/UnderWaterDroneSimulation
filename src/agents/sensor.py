#This is where the sensor agent is defined

from dataclasses import dataclass, field
from typing import List
from ..protocols.dtn_protocol import DTNMessage
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

@dataclass
class Sensor:
    id: str
    position: Position
    messages: List[DTNMessage] = field(default_factory=list)
    data_sequence: int = 0

    def generate_message(self, current_time: float, config: SimulationConfig) -> DTNMessage:
        message = DTNMessage(
            id = f"{self.id}_{self.data_sequence}",
            source_id = self.id,
            destination_id="surface_gateway",
            data=f"sensor_data_{self.data_sequence}",
            generation_time=current_time,
            hop_count=0,
            priority=1,  # Default priority
            ttl=config.message_ttl, #Uses config value
            size=config.message_size #Uses config value
        )
        self.data_sequence += 1
        return message
    
    def add_message_to_buffer(self, message: DTNMessage, config: SimulationConfig) -> bool:
        if len(self.messages) >= config.sensor_buffer_capacity:
            # Dropping oldest message to make space (FIFO queue)
            self.messages.pop(0)
        self.messages.append(message)
        return True

    def get_messages_for_collection(self) -> List[DTNMessage]:
        #Get all messages for collection
        messages_to_send = self.messages.copy()
        self.messages.clear()  # Clear buffer after collection
        return messages_to_send
    
    def has_messages(self) -> bool:
        return len(self.messages) > 0
    
    # Remove manual distance calculation, use Position class:
    def is_in_range(self, other_position: Position, config: SimulationConfig) -> bool:
        # Use the Position class method instead of manual math
        return self.position.is_within_range(other_position, config.sensor_comm_range)
    
    def get_buffer_usage(self, config: SimulationConfig) -> float:
        # Get buffer usage as percentage (0.0 to 1.0)
        return len(self.messages) / config.sensor_buffer_capacity

    def is_buffer_full(self, config: SimulationConfig) -> bool:
        # Check if buffer is at capacity
        return len(self.messages) >= config.sensor_buffer_capacity

    def __str__(self) -> str:
        # String representation for debugging
        return f"Sensor(id={self.id}, messages={len(self.messages)}, pos={self.position})"
        
    