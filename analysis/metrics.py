# analysis/metrics.py
"""
Unified metrics module for DTN simulation, RL training, and offline comparison.

Counts:
- generated: messages created at sensors
- delivered: messages accepted by ship (unique by message id)
- expired: messages removed due to TTL expiry
- dropped: messages removed due to buffer overflow (sensor/drone)
- visits: drone visits to sensors and ship
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EpisodeMetrics:
    aoi_integral: float
    delivered: int
    expired: int
    generated: int
    dropped_sensor_buffer: int
    dropped_drone_buffer: int
    ship_visits: int
    sensor_visits_total: int
    sensor_visits_per_sensor: Dict[int, int] = field(default_factory=dict)

    @property
    def delivery_rate(self) -> float:
        return self.delivered / self.generated if self.generated > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        """Fraction of generated messages that were lost (expired + dropped)."""
        lost = self.expired + self.dropped_sensor_buffer + self.dropped_drone_buffer
        return lost / self.generated if self.generated > 0 else 0.0


@dataclass
class MessageMetrics:
    message_id: str
    source_sensor: int
    generation_time: float
    delivery_time: float
    aoi: float


class MetricsCollector:
    """
    Collects message-based delivery logs + episode-level metrics.

    AoI integral is the exact continuous-time integral of:
        sum_i AoI_i(t)
    where:
        AoI_i(t) = t - last_delivered_generation_time_of_sensor_i
    """

    def __init__(self):
        # Per-message logs (delivered messages only)
        self.messages: List[MessageMetrics] = []

        # Episode-level counters
        self.total_generated: int = 0
        self.total_delivered: int = 0
        self.total_expired: int = 0
        self.total_dropped_sensor_buffer: int = 0
        self.total_dropped_drone_buffer: int = 0

        # Visit tracking
        self.total_ship_visits: int = 0
        self.sensor_visit_counts: Dict[int, int] = {}

        # Continuous-time AoI integral
        self.aoi_integral: float = 0.0
        self.last_aoi_update_time: float = 0.0

        # Per-sensor AoI anchor: last delivered generation time
        self.last_delivery_time_per_sensor: Dict[int, float] = {}

    # -----------------------------------------------------
    # Counters ("hooks")
    # -----------------------------------------------------
    def log_message_generated(self, n: int = 1):
        self.total_generated += int(n)

    def log_message_expired(self, n: int = 1):
        self.total_expired += int(n)

    def log_message_dropped_sensor_buffer(self, n: int = 1):
        self.total_dropped_sensor_buffer += int(n)

    def log_message_dropped_drone_buffer(self, n: int = 1):
        self.total_dropped_drone_buffer += int(n)

    # -----------------------------------------------------
    # Visit tracking
    # -----------------------------------------------------
    def log_ship_visit(self):
        """Log a drone visit to the ship."""
        self.total_ship_visits += 1

    def log_sensor_visit(self, sensor_id: int):
        """Log a drone visit to a sensor."""
        self.sensor_visit_counts[sensor_id] = self.sensor_visit_counts.get(sensor_id, 0) + 1

    # -----------------------------------------------------
    # Delivery logging (ship-side)
    # -----------------------------------------------------
    def log_message_delivery(self, msg, delivery_time: float):
        self.total_delivered += 1
        aoi = float(delivery_time) - float(msg.generation_time)

        self.messages.append(
            MessageMetrics(
                message_id=msg.id,
                source_sensor=int(msg.source_id),
                generation_time=float(msg.generation_time),
                delivery_time=float(delivery_time),
                aoi=aoi,
            )
        )

        # Update AoI anchor: keep freshest delivered update
        sid = int(msg.source_id)
        prev = self.last_delivery_time_per_sensor.get(sid, 0.0)
        self.last_delivery_time_per_sensor[sid] = max(prev, float(msg.generation_time))

    # -----------------------------------------------------
    # Exact continuous-time AoI integral
    # -----------------------------------------------------
    def update_aoi_integral(self, now: float):
        """
        Exact integral over [t0, t1] of sum_i (t - gen_i) dt
        where gen_i is last delivered generation time for sensor i.
        """
        t0 = float(self.last_aoi_update_time)
        t1 = float(now)
        if t1 <= t0:
            return

        dt = t1 - t0
        cost = 0.0

        for gen_time in self.last_delivery_time_per_sensor.values():
            gen = float(gen_time)
            cost += 0.5 * (t1 * t1 - t0 * t0) - gen * dt

        self.aoi_integral += cost
        self.last_aoi_update_time = t1

    # -----------------------------------------------------
    # Finalize episode
    # -----------------------------------------------------
    def finalize(self, final_time: float) -> EpisodeMetrics:
        self.update_aoi_integral(final_time)
        total_sensor_visits = sum(self.sensor_visit_counts.values())
        
        return EpisodeMetrics(
            aoi_integral=self.aoi_integral,
            delivered=self.total_delivered,
            expired=self.total_expired,
            generated=self.total_generated,
            dropped_sensor_buffer=self.total_dropped_sensor_buffer,
            dropped_drone_buffer=self.total_dropped_drone_buffer,
            ship_visits=self.total_ship_visits,
            sensor_visits_total=total_sensor_visits,
            sensor_visits_per_sensor=dict(self.sensor_visit_counts),
        )