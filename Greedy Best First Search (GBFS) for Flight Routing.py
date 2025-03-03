"""
Greedy Best First Search (GBFS) Implementation for Flight Routing

This implementation optimizes flight paths based on air traffic congestion and proximity to destination.
It prioritizes paths with the least congestion and closest proximity to the destination.

Context: Air Traffic Control
Challenge: Define a heuristic for congestion and route proximity and handle sudden air traffic or weather changes
Extension: Handles emergency situations like medical diversions or restricted airspace
"""

import networkx as nx
import matplotlib.pyplot as plt
import heapq
import random

class FlightNode:
    def __init__(self, name, x, y, congestion=0):
        self.name = name
        self.x = x  # x-coordinate (for visualization and distance calculation)
        self.y = y  # y-coordinate
        self.congestion = congestion  # air traffic congestion level (0-10)
    
    def __lt__(self, other):
        # For priority queue comparison
        return True

def euclidean_distance(node1, node2):
    """Calculate straight-line distance between two nodes"""
    return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5

def heuristic(current, goal, congestion_weight=0.5):
    """
    Calculate heuristic value based on:
    1. Distance to goal
    2. Current node's congestion
    
    Lower values are better (less distance, less congestion)
    """
    distance = euclidean_distance(current, goal)
    return distance + (congestion_weight * current.congestion)

def greedy_best_first_search(graph, start, goal):
    """
    Implement Greedy Best First Search algorithm
    Returns the path from start to goal if found, otherwise None
    """
    # Priority queue for open nodes (nodes to visit)
    open_list = [(heuristic(start, goal), start)]
    heapq.heapify(open_list)
    
    # Closed set for visited nodes
    closed_set = set()
    
    # Parent dictionary to reconstruct path
    parent = {start.name: None}
    
    while open_list:
        # Get node with lowest heuristic value
        _, current = heapq.heappop(open_list)
        
        # If goal is reached, reconstruct and return the path
        if current.name == goal.name:
            path = []
            while current:
                path.append(current.name)
                current_name = parent.get(current.name)
                current = next((n for n in graph.nodes() if n.name == current_name), None)
            return path[::-1]  # Reverse path to get from start to goal
        
        # Mark current node as visited
        closed_set.add(current.name)
        
        # Check neighbors
        for neighbor in graph.neighbors(current):
            if neighbor.name in closed_set:
                continue
                
            # If neighbor is not in closed set and not in open list
            if not any(n[1].name == neighbor.name for n in open_list):
                parent[neighbor.name] = current.name
                heapq.heappush(open_list, (heuristic(neighbor, goal), neighbor))
    
    # No path found
    return None

def create_flight_network():
    """Create a flight network with airports as nodes"""
    # Create nodes (airports)
    airports = [
        FlightNode("JFK", 80, 65, congestion=8),   # New York
        FlightNode("LAX", 20, 60, congestion=9),   # Los Angeles
        FlightNode("ORD", 55, 70, congestion=7),   # Chicago
        FlightNode("DFW", 45, 45, congestion=5),   # Dallas
        FlightNode("MIA", 70, 30, congestion=6),   # Miami
        FlightNode("SEA", 15, 85, congestion=4),   # Seattle
        FlightNode("DEN", 40, 60, congestion=3),   # Denver
    ]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes to graph
    for airport in airports:
        G.add_node(airport)
    
    # Add edges (flight routes)
    # Not all airports are connected directly
    G.add_edge(airports[0], airports[2])  # JFK - ORD
    G.add_edge(airports[0], airports[4])  # JFK - MIA
    G.add_edge(airports[1], airports[3])  # LAX - DFW
    G.add_edge(airports[1], airports[5])  # LAX - SEA
    G.add_edge(airports[1], airports[6])  # LAX - DEN
    G.add_edge(airports[2], airports[3])  # ORD - DFW
    G.add_edge(airports[2], airports[6])  # ORD - DEN
    G.add_edge(airports[3], airports[4])  # DFW - MIA
    G.add_edge(airports[3], airports[6])  # DFW - DEN
    G.add_edge(airports[5], airports[6])  # SEA - DEN
    
    return G, airports

def visualize_graph(G, path=None, emergency=None):
    """Visualize the flight network and the selected path"""
    plt.figure(figsize=(12, 8))
    
    # Create positions for nodes based on their coordinates
    pos = {node: (node.x, node.y) for node in G.nodes()}
    
    # Create node labels
    labels = {node: f"{node.name}\nCong: {node.congestion}" for node in G.nodes()}
    
    # Node colors based on congestion (red = high congestion, green = low)
    node_colors = ['#%02x%02x%02x' % (min(255, 50 + 20 * node.congestion), 
                                     max(50, 255 - 20 * node.congestion), 
                                     50) for node in G.nodes()]
    
    # Draw the graph - nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    
    # Highlight emergency node if exists
    if emergency:
        nx.draw_networkx_nodes(G, pos, nodelist=[emergency], 
                              node_color='purple', node_size=800, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5)
    
    # Highlight path edges if provided
    if path and len(path) > 1:
        path_edges = [(next(n for n in G.nodes() if n.name == path[i]), 
                       next(n for n in G.nodes() if n.name == path[i+1])) 
                      for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                              width=3, edge_color='blue', alpha=1.0)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Set title and show the plot
    plt.title("Flight Routing Network - Greedy Best First Search", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    return plt

def handle_emergency(G, current_path, emergency_node, start, goal):
    """Handle emergency situation by rerouting"""
    print(f"\nEMERGENCY at {emergency_node.name}! Rerouting...")
    
    # Temporarily increase congestion at emergency node
    original_congestion = emergency_node.congestion
    emergency_node.congestion = 10  # Maximum congestion
    
    # Find the current position in the path
    current_pos = current_path.index(start.name) if start.name in current_path else 0
    
    # If we've already passed the emergency node, continue on current path
    if emergency_node.name in current_path and current_path.index(emergency_node.name) < current_pos:
        return current_path
    
    # Create a new path avoiding the emergency node
    new_path = greedy_best_first_search(G, start, goal)
    
    # Reset congestion after planning
    emergency_node.congestion = original_congestion
    
    return new_path

def main():
    # Create flight network
    G, airports = create_flight_network()
    
    # Define start and goal airports
    start_airport = next(a for a in airports if a.name == "JFK")
    goal_airport = next(a for a in airports if a.name == "LAX")
    
    print("Airport Network:")
    for a in airports:
        print(f"{a.name}: Congestion level {a.congestion}")
    
    print("\nFinding optimal route from", start_airport.name, "to", goal_airport.name)
    
    # Run Greedy Best First Search
    path = greedy_best_first_search(G, start_airport, goal_airport)
    
    if path:
        print("Optimal path found:", " -> ".join(path))
        
        # Visualize the graph and path
        plt1 = visualize_graph(G, path)
        plt1.savefig('flight_route.png')
        
        # Simulate emergency situation
        emergency_airport = next(a for a in airports if a.name == "DEN")  # Denver has emergency
        
        # Find new path with emergency
        new_start = next(a for a in airports if a.name == path[1])  # Start from second airport in path
        new_path = handle_emergency(G, path, emergency_airport, new_start, goal_airport)
        
        if new_path:
            print("New path after emergency:", " -> ".join(new_path))
            # Visualize with emergency
            plt2 = visualize_graph(G, new_path, emergency=emergency_airport)
            plt2.savefig('flight_route_emergency.png')
        else:
            print("No alternative path found after emergency!")
    else:
        print("No path found from", start_airport.name, "to", goal_airport.name)
        visualize_graph(G).savefig('flight_route_no_path.png')

if __name__ == "__main__":
    main()
