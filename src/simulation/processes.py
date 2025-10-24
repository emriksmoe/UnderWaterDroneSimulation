# This file handles various simulation processes for the application.

#TODO: Make processes handle discrete time steps for drone encounters

import simpy
from typing import List

from ..agents.drone import Drone
from ..agents.ship import Ship
from ..agents.sensor import Sensor
from ..config.simulation_config import SimulationConfig


# Handels sensor processes

def sensor_process(env: simpy.Environment, sensor: Sensor, config: SimulationConfig):
    """Simpy process that generates sensor data at regular intervals."""
    while True:

        message = sensor.generate_message(env.now, config)
        sensor.add_message_to_buffer(message, config)

        print(f"Sensor {sensor.id} generated message at time {env.now}")

        yield env.timeout(config.data_generation_interval)

# Handel drone processes

def drone_process(env: simpy.Environment, drone: Drone, sensors: List[Sensor], ships: List[Ship], 
                  other_drones: List[Drone], config: SimulationConfig):
    """Simpy process that handles drone operations including movement and communication."""
    #At this point only direct movement tis handled, so no encounters with other drones

    while True:

        #Get next target position
        target = drone.get_next_target(sensors, ships, other_drones, config, env.now)

        print(f"Drone {drone.id} moving towards {target.position} at time {env.now}")

        travel_time = drone.calculate_travel_time(target.position, config)

        #Simulate travel time
        yield env.timeout(travel_time)

        #Update drone position
        drone.position = target.position
        print(f"Drone {drone.id} reached {target.position} at time {env.now}")

        preformed_communication = False

        #These checks are based on drone comM_range, note sensor or ship comm_range. But as drone are 
        #currently at exact sensor/ship positions this is sufficient for now
        #TODO: Improve this when discrete time steps are implemented
        #There should be ways to know what exact target we are traveling to, and only check that one.

        #Check for sensors in range to communicate with
        if target.entity_type == "sensor" and target.entity is not None:
            collected = drone.collect_from_sensor(target.entity, config, env.now)
            if collected > 0:
                preformed_communication = True

        #Check for ships in range to communicate with
        if target.entity_type == "ship" and target.entity is not None:
            delivered = drone.deliver_to_ship(target.entity, config, env.now)
            if delivered > 0:
                preformed_communication = True

        #Time spent on operation
        operation_time = config.communication_wait_time if preformed_communication else config.drone_wait_no_action_time
        yield env.timeout(operation_time)

#Ship processes 

def ship_process(env: simpy.Environment, ship: Ship, config: SimulationConfig):
    """SimPy process for ship operations (currently stationary)"""
    while True:
        # Ships are stationary data collection points for now
        # Later you could add patrol patterns or movement
        yield env.timeout(60.0)  # Check every minute for potential operations
            
        # Optional: Print periodic status
        if len(ship.received_messages) > 0:            
            print(f"[{env.now:.1f}s] Ship {ship.id} has {len(ship.received_messages)} messages")

#TODO: Maybe add battery processes later

# Message TTL process

def message_ttl_process(env: simpy.Environment, sensors: List[Sensor], drones: List[Drone], config: SimulationConfig):
    """Process that periodically checks and removes expired messages based on TTL."""
    while True:
        yield env.timeout(config.message_ttl_check_interval)  # Check every 30 seconds

        #Clean up drone buffers
        for drone in drones:
            initial_count = len(drone.messages)
            drone.messages = [msg for msg in drone.messages if (env.now - msg.generation_time) <= msg.ttl]
            removed_count = initial_count - len(drone.messages)
            if removed_count > 0:
                print(f"Drone {drone.id} removed {removed_count} expired messages at time {env.now}")

        #Clean up sensor buffers
        for sensor in sensors:
            initial_count = len(sensor.messages)
            sensor.messages = [msg for msg in sensor.messages if (env.now - msg.generation_time) <= msg.ttl]
            removed_count = initial_count - len(sensor.messages)
            if removed_count > 0:
                print(f"Sensor {sensor.id} removed {removed_count} expired messages at time {env.now}")


#Statistics process

def statistics_process(env: simpy.Environment, sensors: List[Sensor], drones: List[Drone], ships: List[Ship], config: SimulationConfig):
    """Process that periodically collects and prints simulation statistics."""
    while True:
        yield env.timeout(config.statistics_interval)  # Collect stats every defined interval

        total_messages_generated = sum(sensor.data_sequence for sensor in sensors)
        messages_in_sensor_buffers = sum(len(sensor.messages) for sensor in sensors)
        messages_in_drone_buffers = sum(len(drone.messages) for drone in drones)
        total_messages_delivered = sum(len(ship.received_messages) for ship in ships)

        print(f"[{env.now:.1f}s] Statistics:")
        print(f"  Total Messages Generated: {total_messages_generated}")
        print(f"  Total Messages in sensor buffers: {messages_in_sensor_buffers}")
        print(f"  Total Messages in drone buffers: {messages_in_drone_buffers}")
        print(f"  Total Messages Delivered to Ships: {total_messages_delivered}")

        # Calculate delivery ratio
        if total_messages_generated > 0:
            delivery_ratio = total_messages_delivered / total_messages_generated * 100
            print(f"Delivery ratio: {delivery_ratio:.1f}%")
        
        # Per-drone status
        print("\nDrone Status:")
        for drone in drones:
            buffer_usage = len(drone.messages) / config.drone_buffer_capacity * 100
            print(f"  {drone.id}: {len(drone.messages)} messages ({buffer_usage:.1f}% buffer)")
        
        # Per-ship status  
        print("\nShip Status:")
        for ship in ships:
            recent_messages = [msg for msg in ship.received_messages 
                             if env.now - ship.delivery_log[msg.id] <= 300]  # Last 5 minutes
            print(f"  {ship.id}: {len(ship.received_messages)} total, {len(recent_messages)} recent")
        
        print("=" * 50 + "\n")