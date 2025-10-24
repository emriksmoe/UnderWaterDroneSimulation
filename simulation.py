#This file actually runs the simulation environment

from src.simulation.environment import DTNSimulation
from src.config.simulation_config import SimulationConfig

def run_dtn_simulation():
    """Function to run the DTN Simulation"""

    print("Setting up simulation configuration...")
    config = SimulationConfig()

    print(f"Configuration: {config.num_sensors} sensors, {config.num_drones} drones, {config.num_ships} ships")
    print(f"Simulation time: {config.sim_time} seconds\n")

    # Create and run simulation (agents auto-created in constructor)
    simulation = DTNSimulation(config)
    simulation.run()
    
    # Print results
    print("\n=== SIMULATION RESULTS ===")
    results = simulation.get_results()  # You need to add this method
    
    for ship in results["ships"]:
        print(f"{ship.id}: {len(ship.received_messages)} messages received")

if __name__ == "__main__":
    run_dtn_simulation()