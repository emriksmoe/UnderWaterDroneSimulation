"""
Round-Robin Movement Strategy with TSP Optimization
===================================================
Creates optimal tour of all sensors using Nearest Neighbor TSP.
Visits ship after EVERY complete tour (not based on buffer threshold).
Each drone maintains independent tour state for parallel operation.
"""
from typing import List, Optional, Dict
from .movement_strategy import MovementStrategy, TargetResult
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..agents.sensor import Sensor
    from ..agents.drone import Drone
    from ..agents.ship import Ship


class RoundRobinMovementStrategy(MovementStrategy):
    """
    Optimized round-robin movement using shortest path tour.
    
    Policy (BUFFER-BASED):
    1. Pre-compute optimal tour of all sensors (TSP using nearest neighbor)
    2. Each drone follows tour continuously (loops forever)
    3. When buffer ≥80% full → interrupt tour, deliver to ship, resume tour
    4. Never delivers based on "tour completion" - only buffer fullness
    
    This ensures:
    - All sensors are visited regularly in optimal order
    - Deliveries happen based on buffer pressure (realistic)
    - Works well for both uniform and variable generation rates
    - Fair baseline for RL comparison
    """
    
    def __init__(self):
        """Initialize the round-robin strategy"""
        # Tour state per drone (drone_id -> state dict)
        self._drone_states: Dict[int, Dict] = {}
        
        # Computed tour (same for all drones)
        self._tour: Optional[List['Sensor']] = None
        self._tour_initialized: bool = False
    
    def reset(self):
        """Reset all drone states and tour"""
        self._drone_states = {}
        self._tour = None
        self._tour_initialized = False
    
    def _get_drone_state(self, drone_id: int) -> Dict:
        """Get or create state for a specific drone"""
        if drone_id not in self._drone_states:
            self._drone_states[drone_id] = {
                'current_tour_index': 0,      # Current position in tour
                'going_to_ship': False,       # Whether heading to ship for delivery
                'tours_completed': 0,         # Number of complete tours
                'sensors_visited_this_tour': 0  # Sensors visited in current tour
            }
        return self._drone_states[drone_id]
    
    def get_next_target(
        self, 
        drone: 'Drone', 
        sensors: List['Sensor'], 
        ships: List['Ship'],
        config: SimulationConfig,
        current_time: float
    ) -> TargetResult:
        """
        Get next target following round-robin tour.
        
        Logic (BUFFER-BASED DELIVERY):
        1. If tour not initialized → compute TSP tour
        2. If buffer full enough (≥80%) → go to ship immediately
        3. If going_to_ship → go to nearest ship, then resume tour
        4. Otherwise → continue visiting sensors in TSP order (loop forever)
        
        NOTE: No longer delivers after "tour completion" - only when buffer is full.
        This is more efficient for variable lambda rates where some sensors are very active.
        """
        
        # Initialize tour if needed
        if not self._tour_initialized:
            self._initialize_tour(sensors, drone.position)
        
        # Get drone state
        drone_id = int(drone.id.split('_')[-1]) if '_' in drone.id else 0
        state = self._get_drone_state(drone_id)
        
        # ====================================================================
        # BUFFER CHECK: If buffer full enough, deliver to ship
        # ====================================================================
        if drone.is_buffer_full(config):
            state['going_to_ship'] = True
            # Will go to ship on next call
        
        # ====================================================================
        # STATE 1: Going to ship for delivery
        # ====================================================================
        if state['going_to_ship']:
            # Find nearest ship
            if ships:
                nearest_ship = min(ships, key=lambda s: drone.position.distance_to(s.position))
                
                # Reset flag and resume tour where we left off
                state['going_to_ship'] = False
                # Keep current_tour_index - resume tour after delivery
                
                return TargetResult(
                    position=nearest_ship.position,
                    entity_type="ship",
                    entity=nearest_ship
                )
        
        # ====================================================================
        # STATE 2: Continue tour - visit next sensor (loop infinitely)
        # ====================================================================
        # Wrap around if we've completed a full tour
        if state['current_tour_index'] >= len(self._tour):
            state['current_tour_index'] = 0
            state['tours_completed'] += 1
            state['sensors_visited_this_tour'] = 0
        
        # ====================================================================
        # STATE 3: Continue tour - visit next sensor
        # ====================================================================
        if self._tour and state['current_tour_index'] < len(self._tour):
            # Get next sensor in tour
            next_sensor = self._tour[state['current_tour_index']]
            
            # Advance tour index
            state['current_tour_index'] += 1
            state['sensors_visited_this_tour'] += 1
            
            return TargetResult(
                position=next_sensor.position,
                entity_type="sensor",
                entity=next_sensor
            )
        
        # ====================================================================
        # FALLBACK: Go to nearest sensor (shouldn't happen normally)
        # ====================================================================
        if sensors:
            nearest_sensor = min(sensors, key=lambda s: drone.position.distance_to(s.position))
            return TargetResult(
                position=nearest_sensor.position,
                entity_type="sensor",
                entity=nearest_sensor
            )
        
        # Ultimate fallback: stay in place
        return TargetResult(
            position=drone.position,
            entity_type="none",
            entity=None
        )
    
    def _initialize_tour(self, sensors: List['Sensor'], start_position: Position):
        """
        Initialize tour using Nearest Neighbor TSP heuristic.
        
        Builds a tour that visits all sensors in an efficient order
        by always going to the nearest unvisited sensor.
        """
        if not sensors:
            self._tour = []
            self._tour_initialized = True
            return
        
        # Nearest Neighbor TSP
        unvisited = sensors.copy()
        tour = []
        current_pos = start_position
        
        while unvisited:
            # Find nearest unvisited sensor
            nearest = min(unvisited, key=lambda s: current_pos.distance_to(s.position))
            tour.append(nearest)
            current_pos = nearest.position
            unvisited.remove(nearest)
        
        self._tour = tour
        self._tour_initialized = True
        
        print(f"[Round-Robin] Tour initialized with {len(tour)} sensors")
    
    def get_strategy_name(self) -> str:
        return "Round-Robin TSP (Deliver After Tour)"
    
    def get_tour_info(self, drone_id: int = 0) -> Dict:
        """Get information about current tour state for a drone"""
        state = self._get_drone_state(drone_id)
        return {
            'tour_length': len(self._tour) if self._tour else 0,
            'current_index': state['current_tour_index'],
            'sensors_visited_this_tour': state['sensors_visited_this_tour'],
            'tours_completed': state['tours_completed'],
            'going_to_ship': state['going_to_ship']
        }