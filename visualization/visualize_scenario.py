"""
Visualize scenario: 2D & 3D, with and without the Round-Robin path.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.config.simulation_config import SimulationConfig
from src.simulation.agent_factory import AgentFactory
from src.movement.round_robin_movement import RoundRobinMovementStrategy

# Constants
SENSOR_COLOR = "dodgerblue"
SHIP_COLOR = "red"
DRONE_COLOR = "purple"


def plot_2d(sensors, ship_pos, tsp_tour, include_path: bool):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot sensors
    for sensor_id, (x, y, z) in sensors:
        ax.scatter(x, y, c=SENSOR_COLOR, s=120, edgecolors="black", linewidths=1.3,
                   label="Sensor (120s)" if sensor_id == sensors[0][0] else "")
        ax.text(x, y + 15, f"S{sensor_id}", fontsize=8, ha="center")

    # Ship
    ax.scatter(ship_pos[0], ship_pos[1], c=SHIP_COLOR, s=200, marker="s",
               edgecolors="black", linewidths=2, label="Ship")
    ax.text(ship_pos[0], ship_pos[1] + 20, "Ship", fontsize=9, ha="center", fontweight="bold")

    # Optional path
    if include_path and tsp_tour:
        xs = [s.position.x for s in tsp_tour] + [tsp_tour[0].position.x]
        ys = [s.position.y for s in tsp_tour] + [tsp_tour[0].position.y]
        ax.plot(xs, ys, "b--", alpha=0.4, linewidth=1.5, label="Round-Robin Path")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Underwater DTN Scenario - 2D View")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()


def plot_3d(sensors, ship_pos, drone_start_pos, tsp_tour, include_path: bool):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Sensors
    for sensor_id, (x, y, z) in sensors:
        ax.scatter(x, y, z, c=SENSOR_COLOR, s=150, edgecolors="black", linewidths=1.4,
                   label="Sensor (120s)" if sensor_id == sensors[0][0] else "")


    # Drone start
    ax.scatter(drone_start_pos[0] + 30, drone_start_pos[1] + 30, drone_start_pos[2],
               c=DRONE_COLOR, s=350, marker="v", edgecolors="black", linewidths=2.5,
               label="Drone Start", depthshade=False)

    # Ship
    ax.scatter(ship_pos[0], ship_pos[1], ship_pos[2], c=SHIP_COLOR, s=400, marker="s",
               edgecolors="black", linewidths=2, label="Ship", depthshade=False)

    # Optional path
    if include_path and tsp_tour:
        xs = [s.position.x for s in tsp_tour] + [tsp_tour[0].position.x]
        ys = [s.position.y for s in tsp_tour] + [tsp_tour[0].position.y]
        zs = [s.position.z for s in tsp_tour] + [tsp_tour[0].position.z]
        ax.plot(xs, ys, zs, "b--", alpha=0.4, linewidth=1.5, label="Round-Robin Path")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("Underwater DTN Scenario - 3D View")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_zlim(0, 220)
    ax.invert_zaxis()

    plt.tight_layout()
    plt.show()


def visualize_scenario():
    config = SimulationConfig()
    factory = AgentFactory(config)
    sensors, drones, ships = factory.create_all_agents()

    all_sensors = [(s.id, (s.position.x, s.position.y, s.position.z)) for s in sensors]
    ship = ships[0]
    ship_pos = (ship.position.x, ship.position.y, ship.position.z)
    drone_start_pos = (drones[0].position.x, drones[0].position.y, drones[0].position.z)

    # Generate Round-Robin path
    rr = RoundRobinMovementStrategy()
    rr._initialize_tour(sensors, ship.position)
    tsp_tour = rr._tour

    print(f"Visualizing {len(all_sensors)} sensors with uniform generation rate (120s)")
    print(f"Ship at: {ship_pos}")
    print(f"Drone starts at: {drone_start_pos}\n")

    # Show all 4 plots
    plot_2d(all_sensors, ship_pos, tsp_tour, include_path=False)
    plot_3d(all_sensors, ship_pos, drone_start_pos, tsp_tour, include_path=False)
    plot_2d(all_sensors, ship_pos, tsp_tour, include_path=True)
    plot_3d(all_sensors, ship_pos, drone_start_pos, tsp_tour, include_path=True)


if __name__ == "__main__":
    visualize_scenario()