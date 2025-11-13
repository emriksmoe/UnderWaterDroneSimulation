#This is where the drone agent is defined

from dataclasses import dataclass, field
from typing import List, Optional
from ..protocols.dtn_protocol import DTNMessage, DTNProtocol
from ..utils.position import Position
from ..config.simulation_config import SimulationConfig
from ..movement.movement_strategy import MovementStrategy, TargetResult

from .sensor import Sensor
from .ship import Ship

@dataclass
class Drone:
    id: str
    position: Position
    protocol: DTNProtocol
    movement_strategy: MovementStrategy
    messages: List[DTNMessage] = field(default_factory=list)
    battery_level: float = 100.0  # Battery level percentage (might be used in future extensions)
    target_position: Optional[Position] = None  # For movement logic
    last_visited: str = "ship"  # Track last visited entity type

    total_drone_encounters: int = 0  # Track total drone-to-drone encounters
    recent_encounters: dict = field(default_factory=dict)  # drone_id -> last_encounter_time (for cooldowns)

    def calculate_travel_time(self, target: Position, config: SimulationConfig) -> float:
        distance = self.position.distance_to(target)
        return distance / config.drone_speed  # Time = Distance / Speed
    
    def get_next_target(self, sensors: List[Sensor], ships: List[Ship], other_drones: List['Drone'], config: SimulationConfig, current_time: float) -> TargetResult:
        """Determine the next target position based on the movement strategy."""
        return self.movement_strategy.get_next_target(self, sensors, ships, other_drones, config, current_time)

    def can_communicate_with(self, other_position: Position, config: SimulationConfig) -> bool:
        """Check if the drone can communicate with another entity based on distance."""
        return self.position.is_within_range(other_position, config.drone_comm_range)
    
    def add_message(self, message: DTNMessage, config: SimulationConfig, current_time: float) -> bool:
        """Add a message to drones buffer"""
        if len(self.messages) >= config.drone_buffer_capacity:
            return False
        if self.protocol.has_seen_message(message.id):
            return False
        if self.protocol.should_forward(message, self.id, current_time):
            self.messages.append(message)
            self.protocol.mark_seen_message(message.id)
            return True
        return False
    
    # Collecting, Exchanging, and Delivering Messages

    def collect_from_sensor(self, sensor: Sensor, config: SimulationConfig, current_time: float) -> int:
        """Collect messages from a sensor if in range."""
        if not self.can_communicate_with(sensor.position, config):
            return 0
        
        collected_count = 0
        
        #Keep collecting messages until buffer is full or sensor has no more messages
        while (len(self.messages) < config.drone_buffer_capacity and sensor.has_messages()):
            #Get next message from sensor
            message = sensor.get_next_message_for_collection()
            if message is None:
                break

            #Try to add message to drone buffe
            if self.add_message(message, config, current_time):
                collected_count += 1
            else:
                sensor.messages.insert(0, message)  # Put back the message if not added
                break  # Drone buffer full

        if collected_count > 0:
            self.last_visited = "sensor"

        print(f"Drone {self.id} collected {collected_count} messages from Sensor {sensor.id}")
        return collected_count
    

    def exchange_messages_with_drone(self, other_drone: 'Drone', config: SimulationConfig, current_time: float) -> int:
        """Exchange messages with another drone if in range."""
        if not self.can_communicate_with(other_drone.position, config):
            return 0
        
        forward_count = 0
        messages_to_forward = self.protocol.select_messages_to_forward(self.messages, other_drone.id, current_time)

        for message in messages_to_forward:
            if other_drone.protocol.has_seen_message(message.id):
                continue  # Skip already seen messages


            forwarded_message = DTNMessage(
                id=message.id,
                source_id=message.source_id,
                destination_id=message.destination_id,
                data=message.data,
                generation_time=message.generation_time,
                hop_count=message.hop_count + 1,
                ttl=message.ttl,
                priority=message.priority,
                size=message.size
            )
        
            if other_drone.add_message(forwarded_message, config, current_time):
                    forward_count += 1
        
        print(f"Drone {self.id} forwarded {forward_count} messages to Drone {other_drone.id}")
        return forward_count
    

    def deliver_to_ship(self, ship: Ship, config: SimulationConfig, current_time: float) -> int:
        """Deliver messages to a ship if in range."""
        if not self.can_communicate_with(ship.position, config):
            return 0

        delivered_count = len(self.messages)

        #Deliver all messages to the ship
        for message in self.messages:
            ship.receive_message(message, current_time)
        
        self.messages.clear()  # Clear drone's buffer after delivery
        self.last_visited = "ship"

        print(f"Drone {self.id} delivered {delivered_count} messages to Ship {ship.id}")
        return delivered_count
    
    #Methods for discrete time movment
    def update_position_towards_target(self, target_position: Position, time_step: float, speed: float) -> bool:
        """Update position by one time step towards target. Returns True if target reached"""
        distance_per_step = speed * time_step
        distance_to_target = self.position.distance_to(target_position)
        
        if distance_to_target <= distance_per_step:
            # We'll reach the target in this step
            self.position = Position(target_position.x, target_position.y, target_position.z)
            return True
        
        # Move towards target by distance_per_step
        self.position = self.position.move_towards(target_position, distance_per_step)
        return False

    def get_intermediate_position(self, target_position: Position, time_elapsed: float, total_travel_time: float) -> Position:
        """Calculate intermediate position during movement based on time elapsed"""
        if total_travel_time <= 0:
            return self.position
        
        progress = min(time_elapsed / total_travel_time, 1.0)
        return self.position.interpolate_to(target_position, progress)
    
    
    def get_drones_in_encounter_range(self, other_drones: List['Drone'], encounter_range: float) -> List['Drone']:
        """Find all drones within encounter range during movement"""
        nearby_drones = []
        
        for other_drone in other_drones:
            if other_drone.id == self.id:  # Skip self
                continue
                
            distance = self.position.distance_to(other_drone.position)
            if distance <= encounter_range:
                nearby_drones.append(other_drone)
        
        return nearby_drones

    def perform_encounter_communication(self, encountered_drones: List['Drone'], config: SimulationConfig, current_time: float) -> int:
        """Perform unidirectional message exchange with encountered drones (with cooldown)"""
        total_exchanges = 0
        
        for other_drone in encountered_drones:
            # Check if we've recently communicated with this drone
            last_encounter_time = self.recent_encounters.get(other_drone.id, 0.0)
            time_since_last_encounter = current_time - last_encounter_time
            
            if time_since_last_encounter < config.encounter_cooldown_time:
                # Skip this encounter - too recent
                continue
            
            # ONLY ONE-WAY EXCHANGE (this drone forwards to other drone)
            forward_count = self.exchange_messages_with_drone(other_drone, config, current_time)
            
            if forward_count > 0:
                total_exchanges += 1
                
                # Update encounter counters and history for BOTH drones
                self.total_drone_encounters += 1
                other_drone.total_drone_encounters += 1
                
                # Record encounter times for both drones
                self.recent_encounters[other_drone.id] = current_time
                other_drone.recent_encounters[self.id] = current_time
                
                print(f"[{current_time:.1f}s] ENCOUNTER: {self.id} → {other_drone.id} "
                    f"(distance: {self.position.distance_to(other_drone.position):.1f}m, "
                    f"forwarded {forward_count} messages)")
            else:
                print(f"[{current_time:.1f}s] POTENTIAL ENCOUNTER: {self.id} ↔ {other_drone.id} "
                    f"(no new messages to forward)")
        
        return total_exchanges

    
    #Utility Methods

    def get_buffer_usage(self, config: SimulationConfig) -> float:
        """Get buffer usage as percentage (0.0 to 1.0)"""
        return len(self.messages) / config.drone_buffer_capacity

    def is_buffer_full(self, config: SimulationConfig) -> bool:
        """Check if buffer is at capacity"""
        return len(self.messages) >= config.drone_buffer_capacity

    def should_visit_ship(self, config: SimulationConfig) -> bool:
        """Check if drone should prioritize visiting ship"""
        buffer_usage = self.get_buffer_usage(config)
        return (buffer_usage >= 0.9 or  # Buffer 90% full
                (len(self.messages) > 0 and self.last_visited != "ship"))

    def __str__(self) -> str:
        """String representation for debugging"""
        return (f"Drone(id={self.id}, buffer={len(self.messages)}, "
                f"battery={self.battery_level:.1f}%, pos={self.position})")