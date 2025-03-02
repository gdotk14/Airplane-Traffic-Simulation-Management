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
        cost += 1000
        
    # Add airspace fees
    cost += to_point.airspace_fee * 10
    
    return cost

def heuristic(point_id, goal_id, router):
    """
    Heuristic function for RBFS that estimates cost to goal.
    
    Considers:
    1. Direct distance to goal
    2. Country boundary crossings
    3. Known weather patterns
    """
    current = router.points[point_id]
    goal = router.points[goal_id]
    
    # Base heuristic is direct distance to goal
    h = current.distance_to(goal)
    
    # Add estimated cost for country boundary crossings
    if current.country and goal.country and current.country != goal.country:
        # Different countries - potential for fees
        h += 50  # Generic penalty for crossing borders
    
    # Add penalty for areas with known bad weather
    if current.weather_condition > 5:
        h += current.weather_condition * 4
    
    return h

def rbfs_search(router, start_id, goal_id, f_limit=float('inf')):
    """
    Recursive Best-First Search (RBFS) implementation.
    
    Args:
        router: The FlightRouter object
        start_id: Starting point ID
        goal_id: Goal point ID
        f_limit: Upper bound on acceptable solution cost
        
    Returns:
        Tuple of (solution_path, solution_cost, new_f_limit)
    """
    return rbfs_recursive(router, start_id, goal_id, [], f_limit)

def rbfs_recursive(router, current_id, goal_id, path, f_limit):
    """
    Recursive helper function for RBFS.
    
    Args:
        router: The FlightRouter object
        current_id: Current point ID
        goal_id: Goal point ID
        path: Path so far
        f_limit: Upper bound on acceptable solution cost
    
    Returns:
        Tuple of (solution_path, solution_cost, new_f_limit)
    """
    # Add current point to path
    current_path = path + [current_id]
    
    # Check if we've reached the goal
    if current_id == goal_id:
        # Calculate total path cost
        total_cost = 0
        for i in range(len(current_path) - 1):
            from_point = router.points[current_path[i]]
            to_point = router.points[current_path[i + 1]]
            total_cost += calculate_cost(from_point, to_point)
        return current_path, total_cost, f_limit
    
    # Get neighbors (successors)
    successors = []
    for neighbor_id, base_cost in router.get_routes(current_id):
        # Skip if already in path (avoid cycles)
        if neighbor_id in current_path:
            continue
            
        from_point = router.points[current_id]
        to_point = router.points[neighbor_id]
        
        # Calculate g(n) - cost from start to neighbor through current
        g = 0
        for i in range(len(current_path) - 1):
            p1 = router.points[current_path[i]]
            p2 = router.points[current_path[i + 1]]
            g += calculate_cost(p1, p2)
        g += calculate_cost(from_point, to_point)
        
        # Calculate f(n) = g(n) + h(n)
        h = heuristic(neighbor_id, goal_id, router)
        f = g + h
        
        # Add to successors list with f-value
        successors.append((neighbor_id, f, g))
    
    # If no successors, return failure
    if not successors:
        return None, float('inf'), f_limit
    
    # Sort successors by f-value
    successors.sort(key=lambda x: x[1])
    
    # If best successor exceeds f_limit, return failure
    if successors[0][1] > f_limit:
        return None, successors[0][1], f_limit
    
    # Continue search with each successor, starting with the best
    while successors:
        # Get best successor
        best_id, best_f, best_g = successors[0]
        
        # If there's only one successor, set alternative to infinity
        if len(successors) > 1:
            alternative = successors[1][1]
        else:
            alternative = float('inf')
        
        # Recursively search from best successor with new f_limit
        result, cost, new_f_limit = rbfs_recursive(
            router, best_id, goal_id, current_path, min(f_limit, alternative))
        
        # Update f-value of best successor
        successors[0] = (best_id, new_f_limit, best_g)
        
        # Re-sort successors
        successors.sort(key=lambda x: x[1])
        
        # If solution found, return it
        if result:
            return result, cost, new_f_limit
    
    # No solution found after trying all successors
    return None, float('inf'), f_limit

def generate_large_airspace_network(num_points=150, connectivity=0.1):
    """
    Generate a large airspace network for testing.
    
    Args:
        num_points: Number of points in the network
        connectivity: Probability of connection between points (0-1)
    
    Returns:
        FlightRouter object
    """
    router = FlightRouter()
    
    # Define some countries
    countries = ["USA", "Canada", "Mexico", "UK", "France", "Germany", "China", "Japan", "Australia", None]
    
    # Set country fees (higher number = more expensive)
    router.set_country_fee("USA", 1.0)
    router.set_country_fee("Canada", 1.2)
    router.set_country_fee("Mexico", 0.8)
    router.set_country_fee("UK", 1.5)
    router.set_country_fee("France", 1.4)
    router.set_country_fee("Germany", 1.3)
    router.set_country_fee("China", 1.6)
    router.set_country_fee("Japan", 1.7)
    router.set_country_fee("Australia", 1.1)
    
    # Create points with country assignments
    for i in range(num_points):
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        altitude = random.uniform(150, 450)
        
        # Assign to a country (or international airspace)
        country = random.choice(countries)
        
        point = AirspacePoint(i, x, y, altitude, country)
        
        # Set random properties
        point.traffic_density = random.uniform(0, 10)
        point.weather_condition = random.uniform(0, 10)
        point.emergency_level = random.uniform(0, 3)  # Most areas have low emergency levels
        
        # Set airspace fee based on country
        if country and country in router.country_fees:
            point.airspace_fee = router.country_fees[country]
        
        # Randomly set some points as restricted
        if random.random() < 0.05:  # 5% chance of restriction
            point.temporary_restriction = True
            
        router.add_point(point)
    
    # Create routes between points
    for i in range(num_points):
        for j in range(i+1, num_points):
            if random.random() < connectivity:
                distance = router.points[i].distance_to(router.points[j])
                router.add_route(i, j, distance)
    
    # Ensure graph is connected by adding minimum spanning tree edges
    # This ensures there's at least one path between any two points
    visited = {0}  # Start with point 0
    while len(visited) < num_points:
        min_cost = float('inf')
        min_edge = None
        
        for i in visited:
            for j in range(num_points):
                if j not in visited and random.random() < 0.3:  # Check only some edges for efficiency
                    cost = router.points[i].distance_to(router.points[j])
                    if cost < min_cost:
                        min_cost = cost
                        min_edge = (i, j)
        
        if min_edge:
            router.add_route(min_edge[0], min_edge[1], min_cost)
            visited.add(min_edge[1])
        else:
            # No edge found, add a random unvisited point to ensure progress
            unvisited = list(set(range(num_points)) - visited)
            if unvisited:
                visited.add(random.choice(unvisited))
    
    # Create a storm system
    storm_center = random.randint(0, num_points-1)
    storm_radius = random.uniform(50, 150)
    
    for i in range(num_points):
        center = router.points[storm_center]
        point = router.points[i]
        distance = point.distance_to(center)
        
        if distance < storm_radius:
            # Scale weather condition based on distance from storm center
            intensity = 10 * (1 - (distance / storm_radius))
            point.weather_condition = max(point.weather_condition, intensity)
    
    # Create high traffic area
    traffic_center = random.randint(0, num_points-1)
    traffic_radius = random.uniform(40, 100)
    
    for i in range(num_points):
        center = router.points[traffic_center]
        point = router.points[i]
        distance = point.distance_to(center)
        
        if distance < traffic_radius:
            # Scale traffic density based on distance from center
            intensity = 10 * (1 - (distance / traffic_radius))
            point.traffic_density = max(point.traffic_density, intensity)
            
    return router

def simulate_changing_conditions(router, iterations=5, delay=0.5):
    """
    Simulate changing conditions in the airspace over time.
    
    Args:
        router: The FlightRouter object
        iterations: Number of changes to make
        delay: Time delay between changes (seconds)
    
    Returns:
        List of changed point IDs
    """
    changed_points = set()
    
    for _ in range(iterations):
        # Pick a random point to change
        point_id = random.choice(list(router.points.keys()))
        change_type = random.choice(['traffic', 'weather', 'emergency', 'restriction'])
        
        if change_type == 'traffic':
            # Change traffic density
            new_density = random.uniform(0, 10)
            if router.update_traffic(point_id, new_density):
                changed_points.add(point_id)
                print(f"Traffic density at point {point_id} changed to {new_density:.1f}")
        
        elif change_type == 'weather':
            # Change weather condition
            new_condition = random.uniform(0, 10)
            if router.update_weather(point_id, new_condition):
                changed_points.add(point_id)
                print(f"Weather condition at point {point_id} changed to {new_condition:.1f}")
        
        elif change_type == 'emergency':
            # Change emergency level
            new_level = random.uniform(0, 10)
            if router.declare_emergency(point_id, new_level):
                changed_points.add(point_id)
                print(f"Emergency level at point {point_id} changed to {new_level:.1f}")
        
        elif change_type == 'restriction':
            # Toggle restriction
            new_restriction = not router.points[point_id].temporary_restriction
            if router.set_restriction(point_id, new_restriction):
                changed_points.add(point_id)
                status = "RESTRICTED" if new_restriction else "UNRESTRICTED"
                print(f"Point {point_id} is now {status}")
        
        # Sleep to simulate time passing
        time.sleep(delay)
    
    return list(changed_points)

def print_path_details(path, router):
    """Print details about a flight path."""
    if not path:
        print("No path available.")
        return
    
    total_cost = 0
    total_distance = 0
    countries_crossed = set()
    highest_traffic = 0
    worst_weather = 0
    
    print("\nPath details:")
    print("-" * 80)
    print(f"{'Point ID':<8} {'Country':<10} {'Traffic':<8} {'Weather':<8} {'Emergency':<10} {'Fee':<6} {'Status':<12}")
    print("-" * 80)
    
    for i in range(len(path)):
        point = router.points[path[i]]
        
        # Track statistics
        if point.country:
            countries_crossed.add(point.country)
        highest_traffic = max(highest_traffic, point.traffic_density)
        worst_weather = max(worst_weather, point.weather_condition)
        
        # Calculate segment cost
        if i > 0:
            prev_point = router.points[path[i-1]]
            segment_cost = calculate_cost(prev_point, point)
            segment_distance = prev_point.distance_to(point)
            total_cost += segment_cost
            total_distance += segment_distance
        
        # Determine status
        status = []
        if point.temporary_restriction:
            status.append("RESTRICTED")
        if point.emergency_level > 5:
            status.append("EMERGENCY")
        if point.weather_condition > 7:
            status.append("SEVERE WEATHER")
        if point.traffic_density > 7:
            status.append("HEAVY TRAFFIC")
        status_str = ", ".join(status) if status else "Normal"
        
        # Print point details
        print(f"{point.id:<8} {point.country or 'INTL':<10} {point.traffic_density:>6.1f}/10 {point.weather_condition:>6.1f}/10 {point.emergency_level:>8.1f}/10 {point.airspace_fee:>5.1f} {status_str:<12}")
    
    print("-" * 80)
    print(f"Total path cost: {total_cost:.2f}")
    print(f"Total distance: {total_distance:.2f} units")
    print(f"Countries crossed: {', '.join(countries_crossed) if countries_crossed else 'None (international airspace only)'}")
    print(f"Highest traffic density encountered: {highest_traffic:.1f}/10")
    print(f"Worst weather condition encountered: {worst_weather:.1f}/10")

def main():
    """Main function to test the RBFS implementation."""
    # Generate a large airspace network
    print("Generating airspace network...")
    router = generate_large_airspace_network(num_points=150, connectivity=0.15)
    
    # Select random start and goal points
    start_id = random.choice(list(router.points.keys()))
    goal_id = random.choice(list(router.points.keys()))
    while goal_id == start_id:  # Ensure start and goal are different
        goal_id = random.choice(list(router.points.keys()))
    
    print(f"Planning initial flight path from Point {start_id} to Point {goal_id}...")
    
    # Run initial RBFS search
    path, cost, _ = rbfs_search(router, start_id, goal_id)
    
    if path:
        print(f"Initial path found with {len(path)} points and cost {cost:.2f}")
        print(f"Path: {path}")
        print_path_details(path, router)
        
        # Simulate changing conditions
        print("\nSimulating changing conditions...")
        changed_points = simulate_changing_conditions(router, iterations=3, delay=0.5)
        
        # Check if changes affect our path
        affected = any(point_id in path for point_id in changed_points)
        if affected:
            print("\nChanges affect our current path. Rerouting...")
            
            # Find our current position (assume we're at the midpoint for simulation)
            current_index = len(path) // 2
            current_id = path[current_index]
            print(f"Current position: Point {current_id}")
            
            # Run RBFS again from current position
            new_path, new_cost, _ = rbfs_search(router, current_id, goal_id)
            
            if new_path:
                # Combine paths
                complete_path = path[:current_index] + new_path
                print(f"New path found from current position to destination!")
                print(f"Updated path has {len(complete_path)} points")
                print(f"Updated path: {complete_path}")
                print_path_details(complete_path, router)
            else:
                print("Failed to find alternative route to destination.")
        else:
            print("\nChanges do not affect our current path. Continuing as planned.")
    else:
        print("Failed to find initial path to destination.")

if __name__ == "__main__":
    main()
