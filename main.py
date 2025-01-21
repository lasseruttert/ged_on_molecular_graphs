import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque
import random as r
import numpy as np

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

def BUILDNT(graph, root, height, k):
    tree = nx.Graph()
    tree.add_node(root, label=graph.nodes[root]["label"])
    D = {}
    D[root] = 0
    for i in range(1, height):
        F = {}
        for v in list(tree.nodes):
            for u in graph.neighbors(v):
                if u not in D:
                    D[u] = i
                if D[u] + k >= i:
                    tree.add_node(u, label=graph.nodes[u]["label"])
                    tree.add_edge(v, u)
                    F[u] = 1
    return tree

def SDTED(nt1, nt2):
    if nt1 != nt2:
        return 1
    else:
        return 0


def new_SDTED(tree1, tree2):
    return 0

def calculate_GED(graph1, graph2):

    min_GED = float("inf")
    for node1 in graph1.nodes:
        for node2 in graph2.nodes:
            nt1 = create_neighborhood_tree(graph1, node1)
            nt2 = create_neighborhood_tree(graph2, node2)
            GED = SDTED(nt1, nt2)
            if GED < min_GED:
                min_GED = GED

    return min_GED
            

def calculate_cost_matrix(graphs):
    cost_matrix = [[0 for i in range(len(graphs))] for j in range(len(graphs))]
    for i in range(len(graphs)):
        for j in range(len(graphs)):
            cost_matrix[i][j] = calculate_GED(graphs[i+1], graphs[j+1])
            print(t.time() - basetime)

    return np.matrix(cost_matrix)


# print(calculate_cost_matrix(graphs))

# now only for the first 10 graphs
print(calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:10]}))
