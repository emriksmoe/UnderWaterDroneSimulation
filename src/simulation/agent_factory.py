import random
import numpy as np
from typing import List, Tuple, Optional

from ..agents.drone import Drone
from ..agents.sensor import Sensor
from ..agents.ship import Ship
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig
from ..movement.movement_strategy import MovementStrategy
from ..movement.random_movement import RandomMovementStrategy


class AgentFactory:
    """
    Factory class for creating agents with FIXED sensor layout.
    
    All simulations use the SAME sensor positions (seed=42)
    Ensures fair comparison between strategies
    Matches RL training environment EXACTLY
    Uses NumPy RNG (same as RL environment)
    Drones start at ship position (5m deeper)
    """
    
    # CLASS-LEVEL FIXED SCENARIO (shared across all instances)
    _FIXED_SCENARIO_SEED = 42
    _fixed_sensor_positions: Optional[List[Position]] = None
    _fixed_ship_position: Optional[Position] = None

    def __init__(self, config: SimulationConfig, drone_strategy: MovementStrategy = None):
        """
        Initialize agent factory.
        
        Args:
            config: Simulation configuration
            drone_strategy: Movement strategy for drones
        """
        self.config = config
        self.drone_strategy = drone_strategy if drone_strategy else RandomMovementStrategy()
        
        # Generate fixed scenario on first instantiation
        if AgentFactory._fixed_sensor_positions is None:
           # print(f"\n📍 Generating FIXED sensor layout (seed={self._FIXED_SCENARIO_SEED})...")
            self._generate_fixed_scenario()
           # print(f" Fixed layout: {len(AgentFactory._fixed_sensor_positions)} sensors")
            #print(f" Ship position: {AgentFactory._fixed_ship_position}")
           # print(f" First sensor: {AgentFactory._fixed_sensor_positions[0]}")
            #print(f" Last sensor: {AgentFactory._fixed_sensor_positions[-1]}\n")

    @classmethod
    def _generate_fixed_scenario(cls):
        """
        Generate fixed sensor and ship positions (called once).
        
        Uses NumPy RNG (same as RL environment) for exact match
        Enforces min_distance_between_sensors (same as RL environment)
        """
        # Use NumPy random (same as RL environment)
        rng = np.random.RandomState(cls._FIXED_SCENARIO_SEED)
        config = SimulationConfig()  # Use default config for generation
        
        # Generate fixed sensor positions WITH SPACING ENFORCEMENT
        cls._fixed_sensor_positions = []
        max_attempts = 1000
        
        for i in range(config.num_sensors):
            placed = False
            for attempt in range(max_attempts):
                pos = Position(
                    rng.uniform(0, config.area_size[0]),
                    rng.uniform(0, config.area_size[1]),
                    rng.uniform(config.min_depth, config.depth_range)
                )
                
                # Check minimum distance from other sensors
                valid = True
                for existing_pos in cls._fixed_sensor_positions:
                    if pos.distance_to(existing_pos) < config.min_distance_between_sensors:
                        valid = False
                        break
                
                if valid:
                    cls._fixed_sensor_positions.append(pos)
                    placed = True
                    break
            
            if not placed:
                # Fallback: place anyway (shouldn't happen with reasonable constraints)
                pos = Position(
                    rng.uniform(0, config.area_size[0]),
                    rng.uniform(0, config.area_size[1]),
                    rng.uniform(config.min_depth, config.depth_range)
                )
                cls._fixed_sensor_positions.append(pos)
        
        # Fixed ship position (center)
        cls._fixed_ship_position = Position(
            config.area_size[0] / 2,
            config.area_size[1] / 2,
            0.0  # Ships at surface
        )

    def generate_random_position(self, entity_type: str) -> Position:
        """Generate random position for drones (sensors use fixed positions)"""
        x = random.uniform(0, self.config.area_size[0])
        y = random.uniform(0, self.config.area_size[1])

        if entity_type == "sensor":
            z = random.uniform(self.config.min_depth, self.config.depth_range)
        elif entity_type == "drone": 
            z = random.uniform(10, self.config.min_depth)
        elif entity_type == "ship":
            z = 0.0
        else:
            raise ValueError("Unknown entity type for position generation")
        
        return Position(x, y, z)

    def is_too_close(self, new_position: Position, existing_positions: List[Position], min_distance: float):
        """Check if position is too close to existing positions"""
        return any(new_position.distance_to(pos) < min_distance for pos in existing_positions)

    def generate_position_with_constraints(
        self, 
        entity_type: str, 
        existing_positions: List[Position], 
        min_distance: float, 
        max_attempts: int = 100
    ) -> Position:
        """Generate position with distance constraints"""
        for attempt in range(max_attempts):
            position = self.generate_random_position(entity_type)
            if not self.is_too_close(position, existing_positions, min_distance):
                return position
        
        raise RuntimeError(
            f"Could not place {entity_type} without violating distance constraints "
            f"after {max_attempts} attempts."
        )
    
    def create_sensors(self) -> List[Sensor]:
        """
        Create sensors using FIXED positions and VARIABLE generation rates.
        
        Uses class-level fixed positions (same for all strategies)
        Uses sensor-specific generation rates from config
        Positions generated with NumPy RNG (matches RL environment)
        """
        sensors = []
        
        #print(f"Creating {self.config.num_sensors} sensors from FIXED layout...")
        #if self.config.use_variable_sensor_rates:
           # print("  Using VARIABLE generation rates per sensor")
        
        for i, fixed_pos in enumerate(self._fixed_sensor_positions):
            sensor_id = i            
    
            
            sensor = Sensor(
                id=sensor_id,
                position=Position(fixed_pos.x, fixed_pos.y, fixed_pos.z),  #
                generation_interval=self.config.data_generation_interval
            )
            sensors.append(sensor)
            
            # Only print first 3 and last to avoid spam
          #  if i < 3 or i == len(self._fixed_sensor_positions) - 1:
              #  print(f"  {sensor.id} at {sensor.position} (interval: {sensor.generation_interval:.0f}s)")
          #  elif i == 3:
              #  print(f"  ... ({self.config.num_sensors - 4} more sensors)")
        
        return sensors
    
    def create_drones(self) -> List[Drone]:
        """
        Create drones at ship position (5m deeper).
        
        All drones start at ship X/Y coordinates, 5m below surface
        """
        drones = []

       # print(f'Creating {self.config.num_drones} drones...')

        for i in range(self.config.num_drones):
            #  Start at ship position, 5m deeper
            position = Position(
                self._fixed_ship_position.x,
                self._fixed_ship_position.y,
                5.0  # 5 meters below surface
            )

            drone = Drone(
                id=f"drone_{i+1}",
                position=position,
                movement_strategy=self.drone_strategy
            )
            drones.append(drone)
           # print(f"  Drone {drone.id} at ship position (5m deeper): {drone.position}")
        
        return drones
    
    def create_ships(self) -> List[Ship]:
        """
        Create ships using FIXED position.
        
        Uses class-level fixed ship position (center)
        """
        ships = []

      #  print(f'Creating {self.config.num_ships} ships...')

        for i in range(self.config.num_ships):
            #Use fixed ship position
            
            ship = Ship(
                id=f"ship_{i+1}",
                position = Position(
                    self._fixed_ship_position.x,
                    self._fixed_ship_position.y,
                    self._fixed_ship_position.z
                )
            )
            ships.append(ship)
           # print(f"  Ship {ship.id} at FIXED position {ship.position}")
        
        return ships
    
    def create_all_agents(self) -> Tuple[List[Sensor], List[Drone], List[Ship]]:
        """Create all agents (sensors and ship use fixed positions)"""
        sensors = self.create_sensors()  # Fixed positions
        drones = self.create_drones()    # Start at ship (5m deeper)
        ships = self.create_ships()      # Fixed position
        return sensors, drones, ships