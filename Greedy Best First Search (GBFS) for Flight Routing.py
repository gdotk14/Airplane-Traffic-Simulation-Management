"""
Greedy Best First Search (GBFS) Implementation for Flight Routing

This implementation optimizes flight paths based on air traffic congestion and proximity to destination.
It prioritizes paths with the least congestion and closest proximity to the destination.

Context: Air Traffic Control
Challenge: Define a heuristic for congestion and route proximity and handle sudden air traffic or weather changes
Extension: Handles emergency situations like medical diversions or restricted airspace
"""

import heapq
import random
import math

class AirspaceNode:
    """Represents a point in the airspace."""
    def __init__(self, id, x, y, altitude, congestion=0):
        self.id = id
        self.x = x
        self.y = y
        self.altitude = altitude
        self.congestion = congestion  # 0-10 scale of air traffic congestion
        self.restricted = False
        self.weather_severity = 0  # 0-10 scale of weather severity
        
    def distance_to(self, other):
        """Calculate Euclidean distance to another node."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.altitude - other.altitude)**2)
    
    def __str__(self):
        return f"Node {self.id} at ({self.x}, {self.y}, {self.altitude})"

class AirTrafficGraph:
    """Represents the airspace as a graph of connected nodes."""
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.emergency_zones = set()  # Set of node IDs with emergencies
        
    def add_node(self, node):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []
            
    def add_edge(self, node1_id, node2_id, weight=1):
        """Add a bidirectional edge between two nodes."""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.edges[node1_id].append((node2_id, weight))
            self.edges[node2_id].append((node1_id, weight))
    
    def get_neighbors(self, node_id):
        """Get all neighboring nodes of a given node."""
        return self.edges[node_id]
    
    def update_congestion(self, node_id, new_congestion):
        """Update congestion level of a node."""
        if node_id in self.nodes:
            self.nodes[node_id].congestion = new_congestion
    
    def declare_emergency(self, node_id):
        """Declare an emergency at a node (e.g., medical diversion needed)."""
        self.emergency_zones.add(node_id)
    
    def resolve_emergency(self, node_id):
        """Resolve an emergency at a node."""
        if node_id in self.emergency_zones:
            self.emergency_zones.remove(node_id)
    
    def set_restricted(self, node_id, is_restricted=True):
        """Set airspace as restricted or unrestricted."""
        if node_id in self.nodes:
            self.nodes[node_id].restricted = is_restricted
    
    def update_weather(self, node_id, severity):
        """Update weather severity at a node."""
        if node_id in self.nodes:
            self.nodes[node_id].weather_severity = severity

def heuristic(current, goal, graph):
    """
    Heuristic function for GBFS that considers:
    1. Direct distance to goal
    2. Congestion at the current node
    3. Weather severity
    4. Emergency status
    5. Restricted airspace
    
    Lower values are better.
    """
    current_node = graph.nodes[current]
    goal_node = graph.nodes[goal]
    
    # Base heuristic is distance to goal
    h = current_node.distance_to(goal_node)
    
    # Add penalty for congestion (higher congestion = higher cost)
    h += current_node.congestion * 5
    
    # Add penalty for bad weather
    h += current_node.weather_severity * 8
    
    # High penalty for restricted airspace
    if current_node.restricted:
        h += 1000
    
    # Check if node is in an emergency zone
    if current in graph.emergency_zones:
        h += 500  # Add penalty for emergency zones
    
    return h

def greedy_best_first_search(graph, start, goal):
    """
    Implements Greedy Best First Search algorithm for flight routing.
    
    Args:
        graph: The airspace graph
        start: Starting node ID
        goal: Goal node ID
    
    Returns:
        List of node IDs representing the path from start to goal
    """
    # Priority queue for open nodes
    open_set = [(heuristic(start, goal, graph), start)]
    heapq.heapify(open_set)
    
    # Set to track closed nodes
    closed_set = set()
    
    # Dictionary to keep track of parent nodes for path reconstruction
    came_from = {start: None}
    
    while open_set:
        # Get node with lowest heuristic value
        _, current = heapq.heappop(open_set)
        
        # Check if we've reached the goal
        if current == goal:
            # Reconstruct and return the path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse to get path from start to goal
        
        # Add current node to closed set
        closed_set.add(current)
        
        # Explore neighbors
        for neighbor, _ in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue
                
            # If neighbor is not in came_from, it's not yet explored
            if neighbor not in came_from:
                came_from[neighbor] = current
                h = heuristic(neighbor, goal, graph)
                heapq.heappush(open_set, (h, neighbor))
    
    # No path found
    return None

def generate_large_airspace_graph(num_nodes=100, connectivity=0.1):
    """
    Generate a large airspace graph for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
        connectivity: Probability of edge between nodes (0-1)
    
    Returns:
        AirTrafficGraph object
    """
    graph = AirTrafficGraph()
    
    # Create nodes
    for i in range(num_nodes):
        x = random.uniform(0, 1000)  # x-coordinate (nautical miles)
        y = random.uniform(0, 1000)  # y-coordinate (nautical miles)
        altitude = random.uniform(150, 450)  # altitude (100s of feet)
        congestion = random.uniform(0, 10)  # random congestion level
        node = AirspaceNode(i, x, y, altitude, congestion)
        
        # Randomly set some nodes as restricted
        if random.random() < 0.05:  # 5% chance of restricted airspace
            node.restricted = True
            
        # Randomly set weather severity
        node.weather_severity = random.uniform(0, 10)
        
        graph.add_node(node)
    
    # Create edges (connections between nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < connectivity:
                # Calculate distance between nodes as edge weight
                distance = graph.nodes[i].distance_to(graph.nodes[j])
                graph.add_edge(i, j, distance)
    
    # Add some emergency zones
    for _ in range(int(num_nodes * 0.02)):  # 2% of nodes are emergency zones
        emergency_node = random.randint(0, num_nodes-1)
        graph.declare_emergency(emergency_node)
        
    return graph

def main():
    """Main function to test the GBFS implementation."""
    # Generate a large airspace graph
    print("Generating airspace graph...")
    graph = generate_large_airspace_graph(num_nodes=150, connectivity=0.15)
    
    # Select random start and goal nodes
    start_node = random.choice(list(graph.nodes.keys()))
    goal_node = random.choice(list(graph.nodes.keys()))
    while goal_node == start_node:  # Ensure start and goal are different
        goal_node = random.choice(list(graph.nodes.keys()))
    
    print(f"Finding route from Node {start_node} to Node {goal_node}...")
    
    # Run GBFS
    path = greedy_best_first_search(graph, start_node, goal_node)
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path)} nodes")
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(path) - 1):
            node1 = graph.nodes[path[i]]
            node2 = graph.nodes[path[i + 1]]
            total_distance += node1.distance_to(node2)
        
        print(f"Total distance: {total_distance:.2f} units")
        
        # Print details of each node in the path
        print("\nPath details:")
        for node_id in path:
            node = graph.nodes[node_id]
            status = []
            if node.restricted:
                status.append("RESTRICTED")
            if node_id in graph.emergency_zones:
                status.append("EMERGENCY")
            if node.congestion > 7:
                status.append("HIGH CONGESTION")
            if node.weather_severity > 7:
                status.append("SEVERE WEATHER")
            
            status_str = ", ".join(status) if status else "Normal"
            print(f"Node {node_id}: Congestion={node.congestion:.1f}, Weather={node.weather_severity:.1f}, Status={status_str}")
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
