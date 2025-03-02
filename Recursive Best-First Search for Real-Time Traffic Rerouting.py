"""
Recursive Best-First Search (RBFS) Implementation for Real-Time Traffic Rerouting

This implementation adaptively adjusts flight paths based on updated air traffic and weather data.
It focuses on memory efficiency for real-time decision-making while balancing immediate 
rerouting needs with long-term efficiency.

Context: Real-Time Flight Rerouting
Challenge: Optimize memory usage and balance immediate vs. long-term efficiency
Extension: Includes scenarios with flights rerouting through countries with additional airspace fees
"""

import math
import random
import time
from collections import defaultdict

class AirspacePoint:
    """Represents a point in the airspace."""
    def __init__(self, id, x, y, altitude, country=None):
        self.id = id
        self.x = x
        self.y = y
        self.altitude = altitude
        self.country = country  # Country this point belongs to
        self.traffic_density = 0  # 0-10 scale
        self.weather_condition = 0  # 0-10 scale (0 = clear, 10 = severe)
        self.emergency_level = 0  # 0-10 scale
        self.temporary_restriction = False  # Temporary flight restriction
        self.airspace_fee = 0  # Fee for traversing this airspace
        
    def distance_to(self, other):
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.altitude - other.altitude)**2)
    
    def __str__(self):
        return f"Point {self.id} at ({self.x}, {self.y}, {self.altitude}) in {self.country or 'international airspace'}"

class FlightRouter:
    """Manages airspace data and routing algorithms."""
    def __init__(self):
        self.points = {}
        self.routes = defaultdict(list)
        self.countries = set()
        self.country_fees = {}  # Fee multipliers for different countries
        
    def add_point(self, point):
        """Add a point to the airspace."""
        self.points[point.id] = point
        if point.country:
            self.countries.add(point.country)
        
    def add_route(self, from_id, to_id, base_cost=None):
        """
        Add a route between two points.
        
        Args:
            from_id: ID of the first point
            to_id: ID of the second point
            base_cost: Optional base cost (defaults to Euclidean distance)
        """
        if from_id in self.points and to_id in self.points:
            # Calculate default cost based on distance if not provided
            if base_cost is None:
                base_cost = self.points[from_id].distance_to(self.points[to_id])
            
            # Add bidirectional route
            self.routes[from_id].append((to_id, base_cost))
            self.routes[to_id].append((from_id, base_cost))
    
    def get_routes(self, point_id):
        """Get all routes from a point."""
        return self.routes[point_id]
    
    def set_country_fee(self, country, fee_multiplier):
        """Set fee multiplier for a country's airspace."""
        self.country_fees[country] = fee_multiplier
        
        # Update airspace fees for all points in this country
        for point in self.points.values():
            if point.country == country:
                point.airspace_fee = fee_multiplier
    
    def update_traffic(self, point_id, density):
        """Update traffic density at a point."""
        if point_id in self.points:
            old_density = self.points[point_id].traffic_density
            self.points[point_id].traffic_density = density
            return old_density != density
        return False
    
    def update_weather(self, point_id, condition):
        """Update weather condition at a point."""
        if point_id in self.points:
            old_condition = self.points[point_id].weather_condition
            self.points[point_id].weather_condition = condition
            return old_condition != condition
        return False
    
    def declare_emergency(self, point_id, level):
        """Declare emergency at a point with specified severity level."""
        if point_id in self.points:
            old_level = self.points[point_id].emergency_level
            self.points[point_id].emergency_level = level
            return old_level != level
        return False
    
    def set_restriction(self, point_id, is_restricted=True):
        """Set temporary flight restriction at a point."""
        if point_id in self.points:
            old_restriction = self.points[point_id].temporary_restriction
            self.points[point_id].temporary_restriction = is_restricted
            return old_restriction != is_restricted
        return False

def calculate_cost(from_point, to_point):
    """
    Calculate total cost between two points considering all factors.
    
    Returns a cost value where lower is better.
    """
    # Base cost is distance
    cost = from_point.distance_to(to_point)
    
    # Add costs for traffic density
    cost += to_point.traffic_density * 5
    
    # Add costs for weather conditions
    cost += to_point.weather_condition * 8
    
    # Add major penalty for emergency areas
    cost += to_point.emergency_level * 25
    
    # Add extreme penalty for restricted areas
    if to_point.temporary_restriction:
        cost += 1
