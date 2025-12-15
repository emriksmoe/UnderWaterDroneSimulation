# src/agents/sensor.py

from dataclasses import dataclass, field
from typing import List, Optional
from ..protocols.dtn_protocol import DTNMessage
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig


from analysis.metrics import MetricsCollector



@dataclass
class Sensor:
    id: int
    position: Position
    generation_interval: float 
    messages: List[DTNMessage] = field(default_factory=list)
    data_sequence: int = 0
    visit_count: int = 0

    def generate_message(self, current_time: float, config: SimulationConfig) -> DTNMessage:
        message = DTNMessage(
            id=f"{self.id}_{self.data_sequence}",
            source_id=self.id,
            data=f"sensor_data_{self.data_sequence}",
            generation_time=current_time,
            size=config.message_size
        )
        self.data_sequence += 1
        return message

    def add_message_to_buffer(
        self,
        message: DTNMessage,
        config: SimulationConfig,
        metrics: Optional[MetricsCollector] = None,
    ) -> bool:
        """
        FIFO buffer. If full, drop oldest message.
        Option B: count as 'dropped_sensor_buffer' (NOT expired).
        """
        if len(self.messages) >= config.sensor_buffer_capacity:
            self.messages.pop(0)
            if metrics is not None:
                metrics.log_message_dropped_sensor_buffer(1)

        self.messages.append(message)
        return True

    def get_messages_for_collection(self) -> List[DTNMessage]:
        messages_to_send = self.messages.copy()
        self.messages.clear()
        return messages_to_send

    def get_next_message_for_collection(self) -> Optional[DTNMessage]:
        if self.has_messages():
            return self.messages.pop(0)
        return None

    def peek_next_message(self) -> Optional[DTNMessage]:
        if self.has_messages():
            return self.messages[0]
        return None

    def has_messages(self) -> bool:
        return len(self.messages) > 0

    def is_in_range(self, other_position: Position, config: SimulationConfig) -> bool:
        return self.position.is_within_range(other_position, config.sensor_comm_range)

    def get_buffer_usage(self, config: SimulationConfig) -> float:
        return len(self.messages) / config.sensor_buffer_capacity

    def is_buffer_full(self, config: SimulationConfig) -> bool:
        return len(self.messages) >= config.sensor_buffer_capacity

    def __str__(self) -> str:
        return f"Sensor(id={self.id}, messages={len(self.messages)}, pos={self.position})"
