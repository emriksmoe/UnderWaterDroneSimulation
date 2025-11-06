# Metrics calculation and analysis

from typing import List, Dict, Any
from src.agents.sensor import Sensor
from src.agents.drone import Drone
from src.agents.ship import Ship

def calculate_delivery_ratio(sensors: List[Sensor], ships: List[Ship]) -> float:
    """Calculate the message delivery ratio across all sensors and ships."""
    total_generated = sum(sensor.data_sequence for sensor in sensors)
    total_delivered = sum(len(ship.received_messages) for ship in ships)

    return (total_delivered / total_generated) * 100 if total_generated > 0 else 0.0

def calculate_avrage_aoi(ships: List[Ship]) -> float:
    """Calculate avrage Age of Information"""
    all_aoi_data = []

    for ship in ships:
        ship_aoi = ship.get_aoi_data_for_analysis()
        all_aoi_data.extend(ship_aoi)

    if not all_aoi_data:
        return 0.0
    
    return sum(all_aoi_data) / len(all_aoi_data)

