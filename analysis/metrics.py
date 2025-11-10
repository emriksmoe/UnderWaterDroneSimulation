# Metrics calculation and analysis
#TODO: THis file should be looked more into for possible optimizations and improvements

from typing import List, Dict, Any
from src.agents.sensor import Sensor
from src.agents.drone import Drone
from src.agents.ship import Ship

def calculate_delivery_ratio(sensors: List[Sensor], ships: List[Ship]) -> float:
    """Calculate the message delivery ratio across all sensors and ships."""
    total_generated = sum(sensor.data_sequence for sensor in sensors)
    total_delivered = sum(len(ship.received_messages) for ship in ships)

    return (total_delivered / total_generated) * 100 if total_generated > 0 else 0.0

def calculate_average_aoi(ships: List[Ship]) -> float:
    """Calculate avrage Age of Information"""
    all_aoi_data = []

    for ship in ships:
        ship_aoi = ship.get_aoi_data_for_analysis()
        all_aoi_data.extend(ship_aoi)

    if not all_aoi_data:
        return 0.0
    
    return sum(d["age_of_information"] for d in all_aoi_data) / len(all_aoi_data)

def calculate_buffer_utilization(drones: List[Drone], config) -> Dict[str, float]:
    """Calculate drone buffer utilization statistics"""
    if not drones:
        return {"avg": 0.0, "max": 0.0, "min": 0.0}
    
    utilizations = [len(drone.messages) / config.drone_buffer_capacity for drone in drones]
    
    return {
        "avg": sum(utilizations) / len(utilizations) * 100,  # As percentage
        "max": max(utilizations) * 100,
        "min": min(utilizations) * 100
    }

def calculate_network_efficiency(sensors: List[Sensor], drones: List[Drone], ships: List[Ship]) -> Dict[str, Any]:
    """Calculate overall network efficiency metrics"""
    # Total messages in system
    total_generated = sum(sensor.data_sequence for sensor in sensors)
    total_in_transit = sum(len(drone.messages) for drone in drones)
    total_delivered = sum(len(ship.received_messages) for ship in ships)
    
    # Messages lost/expired (if tracking is available)
    total_processed = total_in_transit + total_delivered
    messages_lost = max(0, total_generated - total_processed)
    
    return {
        "total_generated": total_generated,
        "total_in_transit": total_in_transit,
        "total_delivered": total_delivered,
        "messages_lost": messages_lost,
        "loss_ratio": (messages_lost / total_generated * 100) if total_generated > 0 else 0.0
    }

def calculate_latency_metrics(ships: List[Ship]) -> Dict[str, float]:
    """Calculate message latency statistics"""
    all_latencies = []
    
    for ship in ships:
        for message in ship.received_messages:
            if hasattr(message, 'delivery_time') and hasattr(message, 'generation_time'):
                latency = message.delivery_time - message.generation_time
                all_latencies.append(latency)
    
    if not all_latencies:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    
    all_latencies.sort()
    n = len(all_latencies)
    
    return {
        "avg": sum(all_latencies) / n,
        "min": min(all_latencies),
        "max": max(all_latencies),
        "median": all_latencies[n//2] if n % 2 == 1 else (all_latencies[n//2-1] + all_latencies[n//2]) / 2
    }

def calculate_encounter_metrics(drones: List[Drone]) -> Dict[str, Any]:
    """Calculate drone encounter statistics"""
    if not drones:
        return {
            "total_encounters": 0,
            "avg_encounters_per_drone": 0.0,
            "max_encounters": 0,
            "min_encounters": 0,
            "encounter_distribution": []
        }
    
    encounter_counts = [drone.total_drone_encounters for drone in drones]
    total_encounters = sum(encounter_counts) // 2  # Avoid double counting
    
    return {
        "total_encounters": total_encounters,
        "avg_encounters_per_drone": sum(encounter_counts) / len(encounter_counts),
        "max_encounters": max(encounter_counts),
        "min_encounters": min(encounter_counts),
        "encounter_distribution": encounter_counts
    }

def analyze_simulation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Complete analysis of simulation results"""
    sensors = results["sensors"]
    drones = results["drones"] 
    ships = results["ships"]
    config = results.get("config")
    
    analysis = {
        "delivery_ratio": calculate_delivery_ratio(sensors, ships),
        "average_aoi": calculate_average_aoi(ships),
        "total_messages_generated": sum(sensor.data_sequence for sensor in sensors),
        "total_messages_delivered": sum(len(ship.received_messages) for ship in ships),
        "simulation_time": results.get("simulation_time", 0),
        "network_efficiency": calculate_network_efficiency(sensors, drones, ships),
        "latency_metrics": calculate_latency_metrics(ships),
        "encounter_metrics": calculate_encounter_metrics(drones)  # NEW LINE
    }
    
    if config:
        analysis["buffer_utilization"] = calculate_buffer_utilization(drones, config)
    
    return analysis

def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a formatted summary of analysis results"""
    print("\n" + "="*50)
    print("        SIMULATION ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Delivery Ratio:           {analysis['delivery_ratio']:.2f}%")
    print(f"Average AoI:              {analysis['average_aoi']:.2f} seconds")
    print(f"Messages Generated:       {analysis['total_messages_generated']}")
    print(f"Messages Delivered:       {analysis['total_messages_delivered']}")
    print(f"Simulation Time:          {analysis['simulation_time']:.1f} seconds")

    if 'encounter_metrics' in analysis:
        enc = analysis['encounter_metrics']
        print(f"Drone-to-Drone Encounters: {enc['total_encounters']}")
        print(f"Avg Encounters per Drone:   {enc['avg_encounters_per_drone']:.1f}")
    
    if 'buffer_utilization' in analysis:
        buf = analysis['buffer_utilization']
        print(f"Buffer Utilization:       Avg: {buf['avg']:.1f}%, Max: {buf['max']:.1f}%")
    
    if 'network_efficiency' in analysis:
        eff = analysis['network_efficiency']
        print(f"Messages Lost:            {eff['messages_lost']} ({eff['loss_ratio']:.1f}%)")
    
    if 'latency_metrics' in analysis:
        lat = analysis['latency_metrics']
        print(f"Message Latency:          Avg: {lat['avg']:.1f}s, Median: {lat['median']:.1f}s")
    
    print("="*50)

    
