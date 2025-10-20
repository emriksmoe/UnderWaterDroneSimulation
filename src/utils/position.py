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
    
    @classmethod
    def from_tuple(cls, pos: Tuple[float, float, float]) -> 'Position':
        """Create Position from tuple"""
        return cls(pos[0], pos[1], pos[2])
    
    def __str__(self) -> str:
        return f"Position(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f})"
