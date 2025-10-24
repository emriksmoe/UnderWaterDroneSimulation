# Simulation environment setup

import simpy
from typing import List
from .processes import (
    sensor_process,
    drone_process,
    ship_process,
    message_ttl_process,
    statistics_process
)
from ..agents.sensor import Sensor
from ..agents.drone import Drone
from ..agents.ship import Ship
from ..config.simulation_config import SimulationConfig
from .agent_factory import AgentFactory

class DTNSimulation:
    """Main simulation environment"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.env = simpy.Environment()
        self.sensors: List[Sensor] = []
        self.drones: List[Drone] = []
        self.ships: List[Ship] = []
        self.create_agents_with_factory()

    def add_agents(self, sensors: List[Sensor], drones: List[Drone], ships: List[Ship]):
        """Add agents to the simulation"""
        self.sensors = sensors
        self.drones = drones
        self.ships = ships
        print(f"Added {len(sensors)} sensors, {len(drones)} drones, and {len(ships)} ships to the simulation.")
    
    def create_agents_with_factory(self):
        """Create agents using the agent factory"""
        factory = AgentFactory(self.config)
        sensors, drones, ships = factory.create_all_agents()
        self.add_agents(sensors, drones, ships)

    def start_processes(self):
        """Start processes for all agents"""
        print("Starting sensor processes...")

        for sensor in self.sensors:
            self.env.process(sensor_process(self.env, sensor, self.config))

        for drone in self.drones:
            other_drones = [d for d in self.drones if d.id != drone.id]
            self.env.process(drone_process(self.env, drone, self.sensors, self.ships, other_drones, self.config))

        for ship in self.ships:
            self.env.process(ship_process(self.env, ship, self.config))

        #utility processes
        self.env.process(message_ttl_process(self.env, self.sensors, self.drones, self.config))
        self.env.process(statistics_process(self.env, self.sensors, self.drones, self.ships, self.config))

    def run(self):
        """Run the simulation"""
        print("Running simulation...")
        self.start_processes()
        self.env.run(until=self.config.sim_time)
        print("Simulation ended.")
