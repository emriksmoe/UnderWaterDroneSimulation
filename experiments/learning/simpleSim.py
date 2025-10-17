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

SIM_TIME = 3600 # simulation time units, this is 1 hour
AREA_SIZE = (1000, 1000) # area dimensions (x, y)
MAX_DEPTH = 200  # meters (seabed depth)
SURFACE_DEPTH = 0  # meters (ship at surface)
DRONE_SPEED = 2  # units per time, realistic underwater speed in m/s
COMM_RANGE = 100  # communication range
NUKM_SENSORS = 5
SENSOR_DATA_INTERVAL = 60  # seconds between sensor readings

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
        if random.random() < 0.8: #80% chance to go to a sensor
            target_sensor = random.choice(sensors)
            drone.target_position = target_sensor.position
            target_type = "sensor"
        else:
            drone.target_position = ship.position
            target_type = "ship"


        travel_time = calculate_travel_time(drone.position, drone.target_position)
        print(f"Drone {drone.id} traveling to {target_type} at {env.now:.2f}, will take {travel_time:.2f} time units.")

        yield env.timeout(travel_time) #waita until drone arrives