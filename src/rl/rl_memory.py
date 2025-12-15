from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class RLMemory:
    last_ship_visit_time: float = 0.0
    last_sensor_visit_time: Dict[int, float] = field(default_factory=dict)
    last_sensor_pickup_count: Dict[int, int] = field(default_factory=dict)

    # NEW: drone’s belief about what the ship has (updated only at ship)
    last_known_ship_gen_time: Dict[int, float] = field(default_factory=dict)
    last_action: Optional[int] = None

    def reset(self):
        self.last_ship_visit_time = 0.0
        self.last_sensor_visit_time.clear()
        self.last_sensor_pickup_count.clear()
        self.last_known_ship_gen_time.clear()
        self.last_action = None
