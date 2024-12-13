import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque

basetime = t.time()

# Pfad zu den Dateien
path = "G:\Meine Ablage\PG\ged_on_molecular_graphs\data\MUTAG"
dataset_name = "MUTAG"

# Lade Adjacency-Matrix
edges = pd.read_csv(f"{path}\{dataset_name}_A.txt", header=None, sep=",")
edges.columns = ["source", "target"]


# Lade Knoten-zu-Graph-Zuordnung
node_to_graph = pd.read_csv(f"{path}\{dataset_name}_graph_indicator.txt", header=None)
node_to_graph.columns = ["graph_id"]


# Lade Graph-Labels
graph_labels = pd.read_csv(f"{path}\{dataset_name}_graph_labels.txt", header=None)
graph_labels.columns = ["label"]


# Lade Knoten-Labels
node_labels = pd.read_csv(f"{path}\{dataset_name}_node_labels.txt", header=None)
node_labels.columns = ["label"]

# Lade Kanten-Labels
edge_labels = pd.read_csv(f"{path}\{dataset_name}_edge_labels.txt", header=None)
edge_labels.columns = ["label"]

# Erstelle die Graphen
graphs = {}
for graph_id in node_to_graph["graph_id"].unique():
    graph_nodes = node_to_graph[node_to_graph["graph_id"] == graph_id].index + 1
    subgraph_edges = edges[edges["source"].isin(graph_nodes) & edges["target"].isin(graph_nodes)]
    graphs[graph_id] = nx.from_pandas_edgelist(subgraph_edges, source="source", target="target")
    graphs[graph_id].graph["label"] = graph_labels.loc[graph_id - 1, "label"]

    # Füge Knoten-Labels hinzu
    for node in graph_nodes:
        graphs[graph_id].nodes[node]["label"] = node_labels.loc[node - 1, "label"]

    # Füge Kanten-Labels hinzu
    for _, row in subgraph_edges.iterrows():
        source, target = row["source"], row["target"]
        edge_label = edge_labels.loc[edges[(edges["source"] == source) & (edges["target"] == target)].index[0], "label"]
        graphs[graph_id].edges[source, target]["label"] = edge_label
        graphs[graph_id].edges[target, source]["label"] = edge_label  # Ungerichtete Kante (symmetrisch)

print(t.time() - basetime)

# function to create an unfolding neighborhood tree (with redundancies) for a given node
def create_neighborhood_tree(graph, root):
    tree = nx.Graph()
    tree.add_node(root, label=graph.nodes[root]["label"])
    queue = deque([root])
    visited = set([root])

    while queue:
        current_node = queue.popleft()
        for neighbor in graph.neighbors(current_node):
            tree.add_node(neighbor, label= graph.nodes[neighbor]["label"])
            tree.add_edge(current_node, neighbor)

            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return tree

def create_nt_list(graph):
    nt_list = []
    for node in graph.nodes:
        nt_list.append(create_neighborhood_tree(graph, node))
    return nt_list

def SDTED(graph1, graph2):
    return nx.graph_edit_distance(graph1, graph2)
