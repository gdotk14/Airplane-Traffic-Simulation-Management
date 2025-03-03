import matplotlib.pyplot as plt
import networkx as nx
import heapq

# Define the graph as an adjacency list with heuristic costs
class Graph:
    def __init__(self):
        self.graph = {}
        self.heuristics = {}

    def add_edge(self, node1, node2, cost):
        if node1 not in self.graph:
            self.graph[node1] = []
        if node2 not in self.graph:
            self.graph[node2] = []
        self.graph[node1].append((node2, cost))
        self.graph[node2].append((node1, cost))  # Assuming bidirectional paths
    
    def set_heuristic(self, node, h_value):
        self.heuristics[node] = h_value

def a_star(graph, start, goal):
    open_set = [(graph.heuristics[start], 0, start, [])]
    visited = set()
    
    while open_set:
        open_set.sort()
        _, g, current, path = open_set.pop(0)
        
        if current in visited:
            continue
        
        path = path + [current]
        visited.add(current)
        
        if current == goal:
            return path
        
        for neighbor, cost in graph.graph.get(current, []):
            if neighbor not in visited:
                new_g = g + cost
                f_value = new_g + graph.heuristics.get(neighbor, float('inf'))
                open_set.append((f_value, new_g, neighbor, path))
    
    return None

# Creating the graph
g = Graph()
edges = [
    ('A', 'B', 2), ('A', 'C', 5), ('B', 'D', 4), ('B', 'E', 1),
    ('C', 'F', 3), ('D', 'G', 2), ('E', 'G', 3), ('F', 'G', 1)
]

for edge in edges:
    g.add_edge(*edge)

heuristics = {'A': 7, 'B': 6, 'C': 4, 'D': 3, 'E': 2, 'F': 2, 'G': 0}
for node, h in heuristics.items():
    g.set_heuristic(node, h)

start, goal = 'A', 'G'
path = a_star(g, start, goal)
print("Optimal flight path:", path)

# Graph Visualization
G = nx.Graph()
G.add_weighted_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Optimal Flight Path using A*")
plt.show()
