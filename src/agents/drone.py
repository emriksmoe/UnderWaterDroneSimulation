from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from analysis.metrics import MetricsCollector
from ..protocols.dtn_protocol import DTNMessage
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig
from ..movement.movement_strategy import MovementStrategy, TargetResult

from .sensor import Sensor
from .ship import Ship

@dataclass
class Drone:
    id: str
    position: Position
    movement_strategy: MovementStrategy
    messages: List[DTNMessage] = field(default_factory=list)
    target_position: Optional[Position] = None

    def calculate_travel_time(self, target: Position, config: SimulationConfig) -> float:
        distance = self.position.distance_to(target)
        return distance / config.drone_speed
    
    def get_next_target(self, sensors: List[Sensor], ships: List[Ship], config: SimulationConfig, current_time: float) -> TargetResult:
        return self.movement_strategy.get_next_target(self, sensors, ships, config, current_time)

    def can_communicate_with(self, other_position: Position, config: SimulationConfig) -> bool:
        return self.position.is_within_range(other_position, config.drone_comm_range)
    
    def add_message(
        self,
        message: DTNMessage,
        config: SimulationConfig,
        current_time: float,
        metrics=None,
    ) -> bool:
        """
        Add a message to the drone buffer.
        Returns True if accepted, False if buffer full or duplicate.
        """
        # Buffer full → drop
        if len(self.messages) >= config.drone_buffer_capacity:
            if metrics is not None:
                metrics.log_message_dropped_drone_buffer(1)
            return False

        # Duplicate → ignore (not a drop)
        if any(m.id == message.id for m in self.messages):
            return False

        # Accept message
        self.messages.append(message)
        return True

    def collect_from_sensor(self, sensor: Sensor, config: SimulationConfig, current_time: float, metrics=None) -> int:
        """Collect messages from a sensor if in range."""
        if not self.can_communicate_with(sensor.position, config):
            return 0
        
        collected_count = 0
        
        # Keep collecting until buffer full or sensor empty
        while (len(self.messages) < config.drone_buffer_capacity and sensor.has_messages()):
            message = sensor.get_next_message_for_collection()
            if message is None:
                break

            if self.add_message(message, config, current_time, metrics):
                collected_count += 1
            else:
                sensor.messages.insert(0, message)  # Put back if not added
                break

        return collected_count

    def deliver_to_ship(self, ship: Ship, config: SimulationConfig, current_time: float) -> Tuple[int, List[DTNMessage]]:
        """Deliver all messages to ship if in range."""
        if not self.can_communicate_with(ship.position, config):
            return 0, []

        delivered_messages = list(self.messages)

        # Deliver all messages to ship
        for message in delivered_messages:
            ship.receive_message(message, current_time)
        
        self.messages.clear()

        return len(delivered_messages), delivered_messages

    def get_buffer_usage(self, config: SimulationConfig) -> float:
        """Get buffer usage as percentage (0.0 to 1.0)"""
        return len(self.messages) / config.drone_buffer_capacity

    def is_buffer_full(self, config: SimulationConfig) -> bool:
        """Check if buffer is at capacity"""
        return len(self.messages) >= config.drone_buffer_capacity

    def __str__(self) -> str:
        return f"Drone(id={self.id}, buffer={len(self.messages)}, pos={self.position})"