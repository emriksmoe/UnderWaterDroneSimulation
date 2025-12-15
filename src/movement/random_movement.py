import random
from typing import List 
from .movement_strategy import MovementStrategy, TargetResult
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..agents.sensor import Sensor
    from ..agents.drone import Drone
    from ..agents.ship import Ship

class RandomMovementStrategy(MovementStrategy):
    """
    Pure random movement - no intelligence.
    Picks uniformly random target from all sensors and ships.
    Provides baseline for comparison with intelligent strategies.
    """

    def get_next_target(self, drone: 'Drone', sensors: List['Sensor'], ships: List['Ship'],
                        config: SimulationConfig, current_time: float) -> TargetResult:
        
        if drone.is_buffer_full(config):
            return TargetResult(
                position=ships[0].position,
                entity_type="ship",
                entity=ships[0]
            )

        # Pure uniform random over all entities (sensors + ships)
        all_targets = []
        for s in sensors:
            all_targets.append(("sensor", s))
        for sh in ships:
            all_targets.append(("ship", sh))

        if all_targets:
            entity_type, entity = random.choice(all_targets)
            return TargetResult(
                position=entity.position,
                entity_type=entity_type,
                entity=entity
            )

        # Fallback if no targets exist
        return TargetResult(
            position=drone.position,
            entity_type="none",
            entity=None
        )


    def get_strategy_name(self) -> str:
        return "Random Movement Strategy"

