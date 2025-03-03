import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math
import random

class AirportNode:
    def __init__(self, code, name, x, y, congestion=0):
        self.code = code          # Airport code (e.g., JFK)
        self.name = name          # Full airport name
        self.x = x                # x-coordinate (longitude-like)
        self.y = y                # y-coordinate (latitude-like)
        self.congestion = congestion  # Air traffic congestion (0-10)
    
    def __lt__(self, other):
        # For priority queue comparison
        return True
    
    def __str__(self):
        return self.code
    
    def __repr__(self):
        return self.code

class FlightEdge:
    def __init__(self, distance, wind_speed=0, wind_direction=0, restricted=False, weather_penalty=0):
        self.distance = distance          # Distance in nautical miles
        self.wind_speed = wind_speed      # Wind speed in knots
        self.wind_direction = wind_direction  # Wind direction in degrees (0-359)
        self.restricted = restricted      # Whether this airspace is restricted
        self.weather_penalty = weather_penalty  # Additional penalties for storms, etc. (0-10)
    
    def fuel_cost(self, from_node, to_node):
        """Calculate fuel cost considering distance and wind effects"""
        # Basic cost proportional to distance
        cost = self.distance
        
        # Adjust for tailwind/headwind (simplified model)
        # Calculate flight direction
        flight_direction = math.degrees(math.atan2(to_node.y - from_node.y, to_node.x - from_node.x))
        if flight_direction < 0:
            flight_direction += 360
            
        # Calculate wind effect (tailwind reduces cost, headwind increases it)
        wind_effect = self.wind_speed * math.cos(math.radians(flight_direction - self.wind_direction))
        
        # Adjust cost: negative wind_effect = headwind (increases cost)
        #              positive wind_effect = tailwind (decreases cost)
        wind_factor = max(0.7, 1 - (wind_effect / 100))  # Cap benefit at 30% reduction
        cost *= wind_factor
        
        # Add congestion costs
        cost += (from_node.congestion * 10)  # Congestion increases fuel as planes wait/circle
        
        # Add weather penalty (increased impact for more significant effect)
        cost += (self.weather_penalty * 50)  # Increased from 30 to 50 to make weather more impactful
        
        # Add massive penalty for restricted airspace if applicable
        if self.restricted:
            cost += 1000  # Make restricted airspace very expensive but not impossible
            
        return cost

def euclidean_distance(node1, node2):
    """Calculate straight-line distance between two nodes"""
    return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2) ** 0.5

def a_star_search(graph, start, goal, wind_data=None):
    """
    Implement A* search algorithm for flight path planning
    Returns the path and total cost from start to goal if found, otherwise None
    
    Parameters:
    - graph: NetworkX graph with AirportNodes and FlightEdges
    - start: Starting AirportNode
    - goal: Destination AirportNode
    - wind_data: Optional dictionary with wind information
    """
    # Update wind data if provided
    if wind_data:
        for u, v, data in graph.edges(data=True):
            if (u.code, v.code) in wind_data:
                data['object'].wind_speed = wind_data[(u.code, v.code)]['speed']
                data['object'].wind_direction = wind_data[(u.code, v.code)]['direction']
                data['object'].weather_penalty = wind_data[(u.code, v.code)].get('weather_penalty', 0)
    
    # Initialize data structures for A* search
    open_set = []  # Priority queue for nodes to explore
    heapq.heappush(open_set, (0, start))  # (f_score, node)
    
    # Dictionaries to track costs and paths
    g_score = {node: float('inf') for node in graph.nodes()}  # Cost from start to node
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in graph.nodes()}  # Estimated total cost (g + h)
    f_score[start] = euclidean_distance(start, goal)
    
    # For path reconstruction
    came_from = {}
    
    # Set to track visited nodes
    visited = set()
    
    while open_set:
        # Get node with lowest f_score
        current_f, current = heapq.heappop(open_set)
        
        # If we've reached the goal, reconstruct and return the path
        if current == goal:
            path = [current]
            node = current
            total_cost = g_score[current]
            
            while node in came_from:
                node = came_from[node]
                path.append(node)
                
            return list(reversed(path)), total_cost
        
        # Mark as visited
        visited.add(current)
        
        # Check all neighbors
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
                
            # Get the edge data
            edge_data = graph.get_edge_data(current, neighbor)['object']
            
            # Calculate the cost to reach neighbor through current
            tentative_g = g_score[current] + edge_data.fuel_cost(current, neighbor)
            
            if tentative_g < g_score[neighbor]:
                # This is a better path to neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Update f_score with heuristic
                heuristic = euclidean_distance(neighbor, goal)
                f_score[neighbor] = tentative_g + heuristic
                
                # Add to open set if not already there
                if not any(node == neighbor for _, node in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None, None

def create_flight_network():
    """Create a flight network with airports as nodes and routes as edges"""
    # Create airport nodes
    airports = [
        AirportNode("JFK", "John F. Kennedy International", 75, 65, congestion=7),  # New York
        AirportNode("LAX", "Los Angeles International", 10, 60, congestion=9),      # Los Angeles
        AirportNode("ORD", "O'Hare International", 55, 70, congestion=8),           # Chicago
        AirportNode("DFW", "Dallas/Fort Worth International", 45, 45, congestion=6), # Dallas
        AirportNode("MIA", "Miami International", 70, 30, congestion=5),            # Miami
        AirportNode("SEA", "Seattle-Tacoma International", 12, 85, congestion=4),   # Seattle
        AirportNode("DEN", "Denver International", 40, 65, congestion=5),           # Denver
        AirportNode("ATL", "Hartsfield-Jackson Atlanta", 60, 40, congestion=8)      # Atlanta
    ]
    
    # Create NetworkX graph
    G = nx.DiGraph()  # Using directed graph since wind effects differ by direction
    
    # Add nodes to graph
    for airport in airports:
        G.add_node(airport)
    
    # Add edges (flight routes) with distances and initial wind/weather conditions
    # Function to create edges in both directions with potentially different conditions
    def add_bidirectional_edge(from_idx, to_idx, distance, restricted=False):
        from_airport = airports[from_idx]
        to_airport = airports[to_idx]
        
        # Create forward edge
        forward_edge = FlightEdge(
            distance=distance,
            wind_speed=random.randint(0, 50),  # Random wind speed 0-50 knots
            wind_direction=random.randint(0, 359),  # Random direction 0-359 degrees
            restricted=restricted,
            weather_penalty=random.randint(0, 3)  # Initial small weather penalty
        )
        
        # Create reverse edge with different wind conditions
        reverse_edge = FlightEdge(
            distance=distance,
            wind_speed=random.randint(0, 50),
            wind_direction=random.randint(0, 359),
            restricted=restricted,
            weather_penalty=random.randint(0, 3)
        )
        
        # Add edges to graph
        G.add_edge(from_airport, to_airport, object=forward_edge)
        G.add_edge(to_airport, from_airport, object=reverse_edge)
    
    # Add all routes - Modified distances to create more balanced path options
    add_bidirectional_edge(0, 2, 750)  # JFK - ORD
    add_bidirectional_edge(0, 4, 1100)  # JFK - MIA
    add_bidirectional_edge(0, 7, 800)  # JFK - ATL
    add_bidirectional_edge(1, 3, 1250)  # LAX - DFW
    add_bidirectional_edge(1, 5, 950)  # LAX - SEA
    add_bidirectional_edge(1, 6, 850)  # LAX - DEN
    add_bidirectional_edge(2, 3, 800)  # ORD - DFW
    add_bidirectional_edge(2, 6, 900)  # ORD - DEN
    add_bidirectional_edge(3, 4, 1200)  # DFW - MIA
    add_bidirectional_edge(3, 6, 650)  # DFW - DEN
    add_bidirectional_edge(3, 7, 730)  # DFW - ATL
    add_bidirectional_edge(4, 7, 600)  # MIA - ATL
    add_bidirectional_edge(5, 6, 1000)  # SEA - DEN
    add_bidirectional_edge(6, 7, 1200)  # DEN - ATL
    
    # Add direct JFK-DEN connection - new path option for alternative route
    add_bidirectional_edge(0, 6, 1650)  # JFK - DEN
    
    # Add direct ATL-LAX connection - to create a southern path option
    add_bidirectional_edge(7, 1, 1950)  # ATL - LAX
    
    # Add a restricted edge
    add_bidirectional_edge(2, 7, 580, restricted=False)  # ORD - ATL (removed restriction to make it viable)
    
    return G, airports

def generate_weather_event(graph, airports, severity=5):
    """Generate a weather event (storm) affecting several flight paths"""
    # Choose a specific center for the storm to ensure it affects the optimal path
    # Target DEN which is likely on the optimal path from JFK to LAX
    storm_center = next(a for a in airports if a.code == "DEN")
    print(f"Weather event centered near {storm_center.code} ({storm_center.name})")
    
    # Update wind and weather penalties for edges near the storm
    wind_updates = {}
    for u, v, data in graph.edges(data=True):
        # Calculate distance from storm center to this edge's midpoint
        midpoint_x = (u.x + v.x) / 2
        midpoint_y = (u.y + v.y) / 2
        distance_to_storm = math.sqrt((midpoint_x - storm_center.x)**2 + (midpoint_y - storm_center.y)**2)
        
        # Increase radius of storm effect to cover more flight paths
        if distance_to_storm < 35:  # Increased from 30
            edge = data['object']
            
            # More severe for edges connected to Denver
            if u.code == "DEN" or v.code == "DEN":
                # Store original values for visualization
                wind_updates[(u.code, v.code)] = {
                    'speed': min(100, edge.wind_speed + severity * 8),  # Stronger winds near center
                    'direction': (edge.wind_direction + random.randint(-45, 45)) % 360,
                    'weather_penalty': min(10, edge.weather_penalty + severity + 2)  # Higher penalty
                }
            else:
                # Store original values for visualization
                wind_updates[(u.code, v.code)] = {
                    'speed': min(80, edge.wind_speed + severity * 5),
                    'direction': (edge.wind_direction + random.randint(-45, 45)) % 360,
                    'weather_penalty': min(10, edge.weather_penalty + severity)
                }
    
    return wind_updates, storm_center

def visualize_flight_network(G, path=None, storm_center=None, total_cost=None):
    """Visualize the flight network with the optimal path and weather conditions"""
    plt.figure(figsize=(12, 8))
    
    # Create positions for nodes based on their coordinates
    pos = {node: (node.x, node.y) for node in G.nodes()}
    
    # Node colors based on congestion
    node_colors = ['#%02x%02x%02x' % (min(255, 50 + 20 * node.congestion), 
                                     max(50, 255 - 20 * node.congestion), 
                                     50) for node in G.nodes()]
    
    # Edge colors based on weather penalty
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        edge = data['object']
        # Weather-based coloring (darker = worse weather)
        if edge.restricted:
            edge_colors.append('red')  # Restricted airspace
        else:
            # Weather penalty determines color intensity
            weather_intensity = min(255, 100 + edge.weather_penalty * 15)
            edge_colors.append(f'#{weather_intensity:02x}{weather_intensity:02x}{255:02x}')
        
        edge_widths.append(1.0)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    
    # Draw edges
    edges = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    
    # Highlight the optimal path if provided
    if path and len(path) > 1:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=3.0, alpha=1.0)
    
    # Highlight storm center if provided
    if storm_center:
        nx.draw_networkx_nodes(G, pos, nodelist=[storm_center], node_color='purple', node_size=900, alpha=0.7)
    
    # Draw labels
    labels = {node: f"{node.code}\nCong: {node.congestion}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Add wind arrows (simplified)
    for u, v, data in G.edges(data=True):
        edge = data['object']
        if edge.wind_speed > 30:  # Only show significant winds
            # Calculate edge midpoint
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            
            # Convert wind direction to arrow components
            wind_rad = math.radians(edge.wind_direction)
            dx = math.cos(wind_rad) * (edge.wind_speed / 100)
            dy = math.sin(wind_rad) * (edge.wind_speed / 100)
            
            plt.arrow(mid_x, mid_y, dx, dy, head_width=0.5, head_length=0.3, 
                     fc='black', ec='black', alpha=0.5)
    
    # Create legend elements
    plt.plot([], [], 'o', color='green', label='Low Congestion')
    plt.plot([], [], 'o', color='red', label='High Congestion')
    plt.plot([], [], '-', color='red', label='Restricted Airspace')
    plt.plot([], [], '-', color='blue', linewidth=3, label='Optimal Flight Path')
    if storm_center:
        plt.plot([], [], 'o', color='purple', label='Storm Center')
    
    # Add title with path info if available
    if path:
        path_str = ' → '.join([node.code for node in path])
        title = f"Optimal Flight Path: {path_str}"
        if total_cost is not None:
            title += f"\nTotal Cost: {total_cost:.1f} units"
    else:
        title = "Flight Network with Weather and Traffic Conditions"
    
    plt.title(title, fontsize=14)
    plt.legend(loc='best')
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def main():
    """Main function to demonstrate A* search for flight path planning"""
    # Create flight network
    G, airports = create_flight_network()
    
    # Choose start and destination airports
    start_airport = next(a for a in airports if a.code == "JFK")
    destination_airport = next(a for a in airports if a.code == "LAX")
    
    print(f"Finding optimal flight path from {start_airport.code} to {destination_airport.code}")
    
    # Run A* search to find initial optimal path
    initial_path, initial_cost = a_star_search(G, start_airport, destination_airport)
    
    if initial_path:
        path_str = ' → '.join([node.code for node in initial_path])
        print(f"Initial optimal path: {path_str}")
        print(f"Initial cost: {initial_cost:.2f} units")
        
        # Visualize initial path
        plt1 = visualize_flight_network(G, initial_path, total_cost=initial_cost)
        plt1.savefig('initial_flight_path.png')
        
        # Generate a storm/weather event with higher severity
        wind_updates, storm_center = generate_weather_event(G, airports, severity=9)  # Increased severity
        
        # Recalculate optimal path with updated weather conditions
        updated_path, updated_cost = a_star_search(G, start_airport, destination_airport, wind_updates)
        
        if updated_path:
            updated_path_str = ' → '.join([node.code for node in updated_path])
            print(f"\nUpdated optimal path after weather event: {updated_path_str}")
            print(f"Updated cost: {updated_cost:.2f} units")
            
            # Visualize updated path
            plt2 = visualize_flight_network(G, updated_path, storm_center, total_cost=updated_cost)
            plt2.savefig('updated_flight_path.png')
            
            # Compare paths
            if initial_path != updated_path:
                print("\nThe route has changed due to the weather event!")
                
                # Show more details about what changed
                print(f"Original path: {path_str}")
                print(f"New path: {updated_path_str}")
            else:
                print("\nThe route remains the same despite the weather event, but costs may have changed.")
                
            # Calculate cost difference
            cost_diff = updated_cost - initial_cost
            print(f"Cost difference: {cost_diff:.2f} units ({cost_diff/initial_cost*100:.1f}% change)")
        else:
            print("No viable path found after weather event!")
    else:
        print(f"No path found from {start_airport.code} to {destination_airport.code}")
        visualize_flight_network(G).savefig('no_path_found.png')

if __name__ == "__main__":
    main()
    
