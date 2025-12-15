# src/simulation/ship.py

from dataclasses import dataclass, field
from typing import Set, Dict

from ..protocols.dtn_protocol import DTNMessage
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig
from analysis.metrics import MetricsCollector


@dataclass
class Ship:
    id: str
    position: Position
    metrics: MetricsCollector = field(default_factory=MetricsCollector)

    seen_messages: Set[str] = field(default_factory=set)
    delivery_log: Dict[str, float] = field(default_factory=dict)
    visit_count: int = 0

    def has_seen_message(self, message_id: str) -> bool:
        return message_id in self.seen_messages

    def can_communicate_with(self, other_position: Position, config: SimulationConfig) -> bool:
        return self.position.is_within_range(other_position, config.ship_comm_range)

    def receive_message(self, message: DTNMessage, current_time: float) -> bool:
        # Integrate AoI up to delivery time
        self.metrics.update_aoi_integral(current_time)

        if self.has_seen_message(message.id):
            return False

        self.seen_messages.add(message.id)
        self.delivery_log[message.id] = current_time

        self.metrics.log_message_delivery(message, current_time)
        return True

    def get_episode_metrics(self, final_time: float):
        return self.metrics.finalize(final_time)
