"""
A* Search Implementation for Optimal Flight Path

This implementation plans the optimal flight path for an airplane considering distance,
fuel efficiency, and weather conditions. It uses A* search with a heuristic 
that balances multiple factors.

Context: Flight Path Planning
Challenge: Design a heuristic that balances safety, efficiency, and speed
Extension: Includes penalties for detours caused by restricted airspaces or storms
"""

import heapq
import random
import math
from collections import defaultdict

class Waypoint:
    """Represents a waypoint in the airspace."""
    def __init__(self, id, x, y, altitude):
        self.id = id
        self.x = x
        self.y = y
        self.altitude = altitude
        self.wind_speed = 0  # Wind speed in knots
        self.wind_direction = 0  # Wind direction in degrees
        self.air_density = 1.0  # Relative air density (affects fuel efficiency)
        self.restricted = False  # Whether this airspace is restricted
        self.weather_hazard = 0  # 0-10 scale for weather hazards (storms, etc.)
        self.air_traffic_density = 0  # 0-10 scale for air traffic density
        
    def distance_to(self, other):
        """Calculate Euclidean distance to another waypoint."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.altitude - other.altitude)**2)
    
    def __str__(self):
        return f"Waypoint {self.id} at ({self.x}, {self.y}, {self.altitude})"

class AirspaceNetwork:
    """Represents the airspace as a network of connected waypoints."""
    def __init__(self):
        self.waypoints = {}
        self.connections = defaultdict(list)
        
    def add_waypoint(self, waypoint):
        """Add a waypoint to the network."""
        self.waypoints[waypoint.id] = waypoint
        
    def add_connection(self, from_id, to_id, base_cost=None):
        """
        Add a connection between two waypoints.
        
        Args:
            from_id: ID of the first waypoint
            to_id: ID of the second waypoint
            base_cost: Optional cost override (defaults to Euclidean distance)
        """
        if from_id in self.waypoints and to_id in self.waypoints:
            # Calculate default cost based on distance if not provided
            if base_cost is None:
                base_cost = self.waypoints[from_id].distance_to(self.waypoints[to_id])
            
            # Add bidirectional connection
            self.connections[from_id].append((to_id, base_cost))
            self.connections[to_id].append((from_id, base_cost))
    
    def get_connections(self, waypoint_id):
        """Get all connections from a waypoint."""
        return self.connections[waypoint_id]
    
    def update_wind(self, waypoint_id, speed, direction):
        """Update wind data for a waypoint."""
        if waypoint_id in self.waypoints:
            self.waypoints[waypoint_id].wind_speed = speed
            self.waypoints[waypoint_id].wind_direction = direction
    
    def update_weather(self, waypoint_id, hazard_level):
        """Update weather hazard level for a waypoint."""
        if waypoint_id in self.waypoints:
            self.waypoints[waypoint_id].weather_hazard = hazard_level
    
    def set_restricted(self, waypoint_id, is_restricted=True):
        """Set a waypoint as restricted or unrestricted."""
        if waypoint_id in self.waypoints:
            self.waypoints[waypoint_id].restricted = is_restricted
    
    def update_traffic(self, waypoint_id, density):
        """Update air traffic density for a waypoint."""
        if waypoint_id in self.waypoints:
            self.waypoints[waypoint_id].air_traffic_density = density

def calculate_fuel_efficiency(from_waypoint, to_waypoint, aircraft_heading):
    """
    Calculate fuel efficiency factor based on wind, air density, etc.
    
    Returns a value where lower values are more fuel efficient.
    """
    # Extract waypoint data
    wind_speed = to_waypoint.wind_speed
    wind_direction = to_waypoint.wind_direction
    air_density = to_waypoint.air_density
    
    # Calculate wind component (tailwind positive, headwind negative)
    wind_component = wind_speed * math.cos(math.radians(wind_direction - aircraft_heading))
    
    # Basic fuel efficiency calculation
    # Lower values are better (more fuel efficient)
    # - Tailwind improves efficiency (reduces the value)
    # - Higher air density reduces efficiency (increases the value)
    efficiency = 1.0
    efficiency -= wind_component / 100  # Wind effect
    efficiency *= air_density           # Air density effect
    
    # Ensure efficiency stays positive (for pathfinding purposes)
    return max(0.1, efficiency)

def calculate_heading(from_waypoint, to_waypoint):
    """Calculate heading angle from one waypoint to another."""
    dx = to_waypoint.x - from_waypoint.x
    dy = to_waypoint.y - from_waypoint.y
    
    # Calculate heading angle in degrees
    heading = math.degrees(math.atan2(dy, dx))
    if heading < 0:
        heading += 360
    
    return heading

def heuristic(current_id, goal_id, network):
    """
    A* heuristic function that estimates cost to goal.
    
    Considers:
    1. Direct distance to goal
    2. Wind effects on the general path to goal
    3. Weather hazards
    """
    current = network.waypoints[current_id]
    goal = network.waypoints[goal_id]
    
    # Base heuristic is direct distance to goal
    h = current.distance_to(goal)
    
    # Estimate heading to goal for wind calculations
    heading = calculate_heading(current, goal)
    
    # Estimate wind effect based on current waypoint
    wind_component = current.wind_speed * math.cos(math.radians(current.wind_direction - heading))
    wind_factor = 1.0 - (wind_component / 100)  # Tailwind decreases, headwind increases
    h *= max(0.8, min(1.2, wind_factor))  # Limit wind effect
    
    # Add penalty for weather hazards
    h += current.weather_hazard * 10
    
    # Add penalty for restricted airspace
    if current.restricted:
        h += 1000
    
    return h

def a_star_search(network, start_id, goal_id, aircraft_type="medium"):
    """
    A* search implementation for finding optimal flight paths.
    
    Args:
        network: The airspace network
        start_id: ID of the starting waypoint
        goal_id: ID of the goal waypoint
        aircraft_type: Type of aircraft (affects cost calculations)
    
    Returns:
        Tuple of (path, total_cost) or (None, None) if no path is found
    """
    # Priority queue for open nodes - (f_score, waypoint_id)
    open_set = [(0, start_id)]
    
    # Dictionary to track the best path to each waypoint
    came_from = {}
    
    # Cost from start to each waypoint
    g_score = {start_id: 0}
    
    # Estimated total cost from start to goal through each waypoint
    f_score = {start_id: heuristic(start_id, goal_id, network)}
    
    # Set to track waypoints we've fully explored
    closed_set = set()
    
    while open_set:
        # Get waypoint with lowest f_score
        _, current_id = heapq.heappop(open_set)
        
        # Skip if we've already processed this waypoint
        if current_id in closed_set:
            continue
        
        # If we've reached the goal, reconstruct the path
        if current_id == goal_id:
            path = []
            cost = g_score[current_id]
            while current_id:
                path.append(current_id)
                current_id = came_from.get(current_id)
            return path[::-1], cost  # Reverse to get start-to-goal
        
        # Mark current waypoint as processed
        closed_set.add(current_id)
        
        # Process all connections from the current waypoint
        current_waypoint = network.waypoints[current_id]
        for neighbor_id, base_cost in network.get_connections(current_id):
            # Skip if we've already fully explored this neighbor
            if neighbor_id in closed_set:
                continue
            
            neighbor_waypoint = network.waypoints[neighbor_id]
            
            # Calculate true cost including all factors
            heading = calculate_heading(current_waypoint, neighbor_waypoint)
            fuel_efficiency = calculate_fuel_efficiency(current_waypoint, neighbor_waypoint, heading)
            
            # Combine all factors into a single cost
            actual_cost = base_cost
            
            # Adjust for fuel efficiency
            actual_cost *= fuel_efficiency
            
            # Add penalty for air traffic density
            actual_cost += neighbor_waypoint.air_traffic_density * 5
            
            # Add major penalty for weather hazards
            actual_cost += neighbor_waypoint.weather_hazard * 20
            
            # Add extreme penalty for restricted airspace
            if neighbor_waypoint.restricted:
                actual_cost += 2000
                
            # Calculate tentative g_score
            tentative_g_score = g_score[current_id] + actual_cost
            
            # If this path is better than any previous one to this neighbor
            if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                # Update path info
                came_from[neighbor_id] = current_id
                g_score[neighbor_id] = tentative_g_score
                f_score[neighbor_id] = tentative_g_score + heuristic(neighbor_id, goal_id, network)
                
                # Add to open set if not already there
                heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
    
    # No path found
    return None, None

def generate_large_airspace_network(num_waypoints=150, connectivity=0.1):
    """
    Generate a large airspace network for testing.
    
    Args:
        num_waypoints: Number of waypoints in the network
        connectivity: Probability of connection between waypoints (0-1)
    
    Returns:
        AirspaceNetwork object
    """
    network = AirspaceNetwork()
    
    # Create waypoints
    for i in range(num_waypoints):
        x = random.uniform(0, 1000)  # x-coordinate (nautical miles)
        y = random.uniform(0, 1000)  # y-coordinate (nautical miles)
        altitude = random.uniform(150, 450)  # altitude (100s of feet)
        waypoint = Waypoint(i, x, y, altitude)
        
        # Set random properties
        waypoint.wind_speed = random.uniform(0, 50)  # 0-50 knots
        waypoint.wind_direction = random.uniform(0, 360)  # 0-360 degrees
        waypoint.air_density = random.uniform(0.8, 1.2)  # Relative density
        waypoint.weather_hazard = random.uniform(0, 10)  # Hazard level
        waypoint.air_traffic_density = random.uniform(0, 10)  # Traffic density
        
        # Randomly set some waypoints as restricted
        if random.random() < 0.05:  # 5% chance of restricted airspace
            waypoint.restricted = True
            
        network.add_waypoint(waypoint)
    
    # Create connections
    for i in range(num_waypoints):
        for j in range(i+1, num_waypoints):
            # Add connection with probability based on connectivity parameter
            if random.random() < connectivity:
                # Calculate distance as base cost
                distance = network.waypoints[i].distance_to(network.waypoints[j])
                network.add_connection(i, j, distance)
    
    # Create a storm system (cluster of waypoints with high weather hazard)
    storm_center = random.randint(0, num_waypoints-1)
    storm_radius = random.uniform(50, 150)
    storm_intensity = random.uniform(7, 10)
    
    for i in range(num_waypoints):
        waypoint = network.waypoints[i]
        center = network.waypoints[storm_center]
        distance = waypoint.distance_to(center)
        
        if distance < storm_radius:
            # Scale intensity based on distance from storm center
            intensity_factor = 1 - (distance / storm_radius)
            waypoint.weather_hazard = max(waypoint.weather_hazard, storm_intensity * intensity_factor)
            
    return network

def main():
    """Main function to test the A* implementation."""
    # Generate a large airspace network
    print("Generating airspace network...")
    network = generate_large_airspace_network(num_waypoints=150, connectivity=0.15)
    
    # Select random start and goal waypoints
    start_id = random.choice(list(network.waypoints.keys()))
    goal_id = random.choice(list(network.waypoints.keys()))
    while goal_id == start_id:  # Ensure start and goal are different
        goal_id = random.choice(list(network.waypoints.keys()))
    
    print(f"Finding optimal flight path from Waypoint {start_id} to Waypoint {goal_id}...")
    
    # Run A* search
    path, cost = a_star_search(network, start_id, goal_id)
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path)} waypoints")
        print(f"Total cost: {cost:.2f}")
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(path) - 1):
            waypoint1 = network.waypoints[path[i]]
            waypoint2 = network.waypoints[path[i + 1]]
            total_distance += waypoint1.distance_to(waypoint2)
        
        print(f"Total distance: {total_distance:.2f} nautical miles")
        
        # Print details of each waypoint in the path
        print("\nPath details:")
        for waypoint_id in path:
            waypoint = network.waypoints[waypoint_id]
            status = []
            if waypoint.restricted:
                status.append("RESTRICTED")
            if waypoint.weather_hazard > 7:
                status.append("STORM")
            if waypoint.air_traffic_density > 7:
                status.append("HIGH TRAFFIC")
            
            status_str = ", ".join(status) if status else "Normal"
            print(f"Waypoint {waypoint_id}: Wind={waypoint.wind_speed:.1f}kts/{waypoint.wind_direction:.0f}°, " +
                  f"Weather={waypoint.weather_hazard:.1f}, Traffic={waypoint.air_traffic_density:.1f}, Status={status_str}")
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
