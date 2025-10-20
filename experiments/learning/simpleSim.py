#Basic first simulation of the DTN system only using a few sensors, 1 drone and 1 ship.
#In this simulation the drone moves randomly between sensors, and chooses to go to the ship at random intervals to deliver data.
#Placement of sensors is random within the area.
#Ship is stationary at a fixed location in the area.

import simpy
import random
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Optional

#Simulation parameters, these are just a guess

SIM_TIME = 86400  # simulation time units, this is 24 hours
NUM_SIMULATIONS = 100  # number of simulation runs for averaging results
AREA_SIZE = (1000, 1000) # area dimensions (x, y)
MAX_DEPTH = 200  # meters (seabed depth)
SURFACE_DEPTH = 0  # meters (ship at surface)
DRONE_SPEED = 2  # units per time, realistic underwater speed in m/s
COMM_RANGE = 100  # communication range
NUM_SENSORS = 5
SENSOR_DATA_INTERVAL = 60  # seconds between sensor readings
VISIT_DURATION = 10  # time spent at each sensor or ship
VISIT_PROBABILITY = 0.8  # probability of visiting sensor

# Data Classes

@dataclass
class Position:
    x: float
    y: float
    z: float  # depth, 0 is surface, positive downwards

@dataclass
class Message:
    id: int
    data: str
    generation_time : float #When data was generated at sensor
    timestamp: float #When data was sent
    sender_id: int
    size: int = 100 # size in bytes

    def get_age_of_information(self, current_time: float) -> float: #Gives you AoI
        return current_time - self.generation_time

@dataclass
class Sensor:
    id: int
    position: Position
    messages: List[Message]
    last_data_time: float = 0
    data_sequence: int = 0

@dataclass
class Drone:
    id: int
    position: Position
    buffer_messages: List[Message]
    battery_level: float = 100.0  # percentage
    target_position: Optional[Position] = None
    last_visited: str = "ship"  # Track what was visited last: "sensor" or "ship"

@dataclass
class Ship:
    id: int
    position: Position
    received_messages: List[Message]

@dataclass
class SimulationMetrics:
    delivered_messages: List[Message]
    generation_times: List[float]
    delivery_times: List[float]

    def __init__(self):
        self.delivered_messages = []
        self.generation_times = []
        self.delivery_times = []

    def record_delivery(self, message: Message, delivery_time: float):
        self.delivered_messages.append(message)
        self.delivery_times.append(delivery_time)

    def get_mean_aoi(self) -> float:
        if not self.delivered_messages:
            return 0.0
        total_aoi = sum(
            message.get_age_of_information(delivery_time)
            for message, delivery_time in zip(self.delivered_messages, self.delivery_times)
        )
        return total_aoi / len(self.delivered_messages)


# Helper functionns

def calculate_distance(pos1: Position, pos2: Position) -> float: #Gets distance between two 3D points
    """Calculate 3D Euclidean distance between two positions"""
    return math.sqrt(
        (pos1.x - pos2.x)**2 +
        (pos1.y - pos2.y)**2 +
        (pos1.z - pos2.z)**2
    )

def in_communication_range(pos1: Position, pos2: Position) -> bool: #Checks if two positions are within communication range
    """Check if two positions are within communication range"""
    return calculate_distance(pos1, pos2) <= COMM_RANGE

def calculate_travel_time(pos1: Position, pos2: Position) -> float:
    """Calculate travel time between two positions based on drone speed"""
    distance = calculate_distance(pos1, pos2)
    return distance / DRONE_SPEED


#Simpy processes

def sensor_process(env, sensor: Sensor):
    while True:
        #Genrate new data message
        message = Message(
            id=f"{sensor.id}-{sensor.data_sequence}",
            data=f"Data from sensor {sensor.id}",
            generation_time=env.now,  # This value may not be needed
            timestamp=env.now,
            sender_id=sensor.id
        )

        sensor.messages.append(message)
        sensor.data_sequence += 1
        sensor.last_data_time = env.now

        yield env.timeout(SENSOR_DATA_INTERVAL) #Wait until next data generation

def drone_process(env, drone: Drone, sensors: List[Sensor], ship: Ship, metrics: SimulationMetrics):

    while True:
        #Chooses a next random target, either a sensor or the ship
        if drone.last_visited == "ship" or random.random() < VISIT_PROBABILITY:
            target_sensor = random.choice(sensors)
            drone.target_position = target_sensor.position
            target_type = "sensor"
        else:
            drone.target_position = ship.position
            target_type = "ship"

        travel_time = calculate_travel_time(drone.position, drone.target_position)
        #print(f"Drone {drone.id} traveling to {target_type} at {env.now:.2f}, will take {travel_time:.2f} time units (seconds).")

        yield env.timeout(travel_time) #wait until drone arrives
        drone.position = drone.target_position  # Update drone position

        if target_type == 'ship':
            #Handles delivery of messages to the ship
            for message in drone.buffer_messages:
                ship.received_messages.append(message)
                metrics.record_delivery(message, env.now)
                #print(f"Drone {drone.id} delivered message {message.id} to ship at {env.now:.2f}.")
            drone.buffer_messages.clear()  # Clear buffer after delivery
            #print(f"Drone {drone.id} cleared buffer after delivery at {env.now:.2f}.")
            drone.last_visited = "ship"
        else:
            #Handles collection of messages from the sensor
            for sensor in sensors:
                if in_communication_range(drone.position, sensor.position):
                    collected_count = len(sensor.messages)
                    drone.buffer_messages.extend(sensor.messages)
                    sensor.messages.clear()  # Clear messages after collection
                    #print(f"Drone {drone.id} collected {collected_count} messages from sensor {sensor.id} at {env.now:.2f}.")
            drone.last_visited = "sensor"
        yield env.timeout(VISIT_DURATION)  # Wait a bit before next action


        #Should probably implement some battery management here and DTN routing logic, but for now we just let the drone do its thing.


# Main simulation setup

def run_single_simulation(run_number=None):
    #Run single simulation
    env = simpy.Environment()
    metrics = SimulationMetrics()

    #creating ship
    ship = Ship(
        id = 1,
        position = Position(
            random.uniform(0, AREA_SIZE[0]),
            random.uniform(0, AREA_SIZE[1]),
            0,
        ),
        received_messages=[]
    )

    #creating sensors
    sensors = []
    for i in range(NUM_SENSORS):
        pos = Position(
            random.uniform(0, AREA_SIZE[0]),
            random.uniform(0, AREA_SIZE[1]),
            MAX_DEPTH
        )
        sensors.append(Sensor(id=i, position=pos, messages=[]))

    #creating drone, starting at random position
    drone = Drone(
        id=1,
        position=Position(
            random.uniform(0, AREA_SIZE[0]),
            random.uniform(0, AREA_SIZE[1]),
            random.uniform(50, MAX_DEPTH)
        ),
        buffer_messages=[],
        last_visited="ship"
    )

    #Start all processes

    for sensor in sensors:
        env.process(sensor_process(env, sensor))

    env.process(drone_process(env, drone, sensors, ship, metrics))

    #print(f"Starting simulation for {SIM_TIME} time units (seconds).")
    env.run(until=SIM_TIME)
    #print("Simulation ended.")
    print(f"Simulation Run {run_number}")

    # print(f"\n=== Simulation Results ===")
    #print(f"Total messages delivered: {len(metrics.delivered_messages)}")
    #print(f"Messages still in drone buffer: {len(drone.buffer_messages)}")
    #print(f"Messages still at sensors: {sum(len(s.messages) for s in sensors)}")
   # print(f"Mean Age of Information: {metrics.get_mean_aoi():.2f} time units")
    return {
        'messages_delivered': len(metrics.delivered_messages),
        'messages_in_buffer': len(drone.buffer_messages),
        'messages_at_sensors': sum(len(s.messages) for s in sensors),
        'mean_aoi': metrics.get_mean_aoi(),
        'total_messages_generated': sum(s.data_sequence for s in sensors),
        'delivery_ratio': len(metrics.delivered_messages) / max(1, sum(s.data_sequence for s in sensors))
    }
def run_multiple_simulation(number_of_simulations: int):
    print(f"Running {number_of_simulations} simulations...")

    all_results = []

    for i in range(number_of_simulations):
        # Set different random seed for each run to ensure variety
        random.seed(i * 42)  # Different seed each time
        np.random.seed(i * 42)
        
        result = run_single_simulation(run_number=i+1)
        all_results.append(result)

            # Calculate aggregate statistics
    messages_delivered = [r['messages_delivered'] for r in all_results]
    messages_in_buffer = [r['messages_in_buffer'] for r in all_results]
    messages_at_sensors = [r['messages_at_sensors'] for r in all_results]
    mean_aois = [r['mean_aoi'] for r in all_results if r['mean_aoi'] > 0]  # Only non-zero AoI
    total_generated = [r['total_messages_generated'] for r in all_results]
    delivery_ratios = [r['delivery_ratio'] for r in all_results]
    
    print(f"\n=== AGGREGATE RESULTS FROM {number_of_simulations} SIMULATIONS ===")
    print(f"Messages Delivered:")
    print(f"  Mean: {np.mean(messages_delivered):.2f}")
    print(f"  Std:  {np.std(messages_delivered):.2f}")
    print(f"  Min:  {np.min(messages_delivered)}")
    print(f"  Max:  {np.max(messages_delivered)}")
    
    print(f"\nMessages Generated:")
    print(f"  Mean: {np.mean(total_generated):.2f}")
    print(f"  Std:  {np.std(total_generated):.2f}")
    
    print(f"\nDelivery Ratio:")
    print(f"  Mean: {np.mean(delivery_ratios):.3f}")
    print(f"  Std:  {np.std(delivery_ratios):.3f}")
    print(f"  Min:  {np.min(delivery_ratios):.3f}")
    print(f"  Max:  {np.max(delivery_ratios):.3f}")
    
    if mean_aois:
        print(f"\nAge of Information (only simulations with deliveries):")
        print(f"  Mean: {np.mean(mean_aois):.2f} time units")
        print(f"  Std:  {np.std(mean_aois):.2f}")
        print(f"  Min:  {np.min(mean_aois):.2f}")
        print(f"  Max:  {np.max(mean_aois):.2f}")
    else:
        print(f"\nNo messages delivered in any simulation!")
    
    print(f"\nUndelivered Messages:")
    print(f"  Mean in buffer: {np.mean(messages_in_buffer):.2f}")
    print(f"  Mean at sensors: {np.mean(messages_at_sensors):.2f}")
    
    return all_results

if __name__ == "__main__":
    run_multiple_simulation(NUM_SIMULATIONS)