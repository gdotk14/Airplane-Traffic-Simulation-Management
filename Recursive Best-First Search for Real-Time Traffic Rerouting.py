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

def rbfs(graph, current, goal, f_limit):
    if current == goal:
        return [current], 0

    successors = []
    for neighbor, cost in graph.graph.get(current, []):
        f_value = cost + graph.heuristics.get(neighbor, float('inf'))
        successors.append((f_value, neighbor, cost))
    
    if not successors:
        return None, float('inf')

    successors.sort()
    while successors:
        best_f, best_node, best_g = successors[0]
        if best_f > f_limit:
            return None, best_f

        alternative_f = successors[1][0] if len(successors) > 1 else float('inf')
        result, new_f = rbfs(graph, best_node, goal, min(f_limit, alternative_f))
        successors[0] = (new_f, best_node, best_g)
        successors.sort()
        if result:
            return [current] + result, new_f
    
    return None, float('inf')

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
path, _ = rbfs(g, start, goal, float('inf'))
print("Best path:", path)

# Graph Visualization
G = nx.Graph()
G.add_weighted_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Flight Rerouting Graph")
plt.show()
