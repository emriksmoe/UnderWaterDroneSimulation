#This file defines position utilities that are used across the simulation

from dataclasses import dataclass
import math
from typing import Tuple

@dataclass
class Position:
    x: float
    y: float
    z: float  # Depth, positive downwards

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2 +
                         (self.z - other.z) ** 2)

    def is_within_range(self, other: 'Position', range_limit: float) -> bool:
        """Check if another position is within a certain range."""
        return self.distance_to(other) <= range_limit
    


    def as_tuple(self) -> Tuple[float, float, float]:
        """Return position as a tuple."""
        return (self.x, self.y, self.z)
    
    def interpolate_to(self, other: 'Position', progress: float) -> 'Position':
        """Interpolate between this position and another based on progress [0,1]."""
        progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1

        new_x = self.x + (other.x - self.x) * progress
        new_y = self.y + (other.y - self.y) * progress
        new_z = self.z + (other.z - self.z) * progress

        return Position(new_x, new_y, new_z)
    
    def move_towards(self, target: 'Position', distance: float) -> 'Position':
        """Move from current position towards target by specified distance"""
        current_distance = self.distance_to(target)

        if current_distance <= distance:
            return Position(target.x, target.y, target.z)  # We can reach target
        
        dx = target.x - self.x
        dy = target.y - self.y  
        dz = target.z - self.z
        
    
        # Normalize by current distance
        dx /= current_distance
        dy /= current_distance
        dz /= current_distance
        
        # Move by specified distance
        return Position(
            self.x + dx * distance,
            self.y + dy * distance,
            self.z + dz * distance
        )
    

    def get_direction_to(self, target: 'Position') -> Tuple[float, float, float]:
        """Get normalized direction vector to target position"""
        distance = self.distance_to(target)
        if distance == 0:
            return (0.0, 0.0, 0.0)
        
        return (
            (target.x - self.x) / distance,
            (target.y - self.y) / distance,
            (target.z - self.z) / distance
        )
            


    
    @classmethod
    def from_tuple(cls, pos: Tuple[float, float, float]) -> 'Position':
        """Create Position from tuple"""
        return cls(pos[0], pos[1], pos[2])
    
    def __str__(self) -> str:
        return f"Position(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f})"
