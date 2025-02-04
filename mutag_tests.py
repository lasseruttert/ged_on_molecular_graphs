import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque
import random as r
import numpy as np
from scipy.optimize import linear_sum_assignment
import main as main

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
    # gib jeden Graphen eine ID
    graphs[graph_id].graph["id"] = graph_id

    # Füge Knoten-Labels hinzu
    for node in graph_nodes:
        graphs[graph_id].nodes[node]["label"] = node_labels.loc[node - 1, "label"]

    # Füge Kanten-Labels hinzu
    for _, row in subgraph_edges.iterrows():
        source, target = row["source"], row["target"]
        edge_label = edge_labels.loc[edges[(edges["source"] == source) & (edges["target"] == target)].index[0], "label"]
        graphs[graph_id].edges[source, target]["label"] = edge_label
        graphs[graph_id].edges[target, source]["label"] = edge_label  # Ungerichtete Kante (symmetrisch)

def print_two_graphs(graph1, graph2, layout='spring'):
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))

    if layout == 'spring':
        pos1 = nx.spring_layout(graph1)
        pos2 = nx.spring_layout(graph2)
    elif layout == 'circular':
        pos1 = nx.circular_layout(graph1)
        pos2 = nx.circular_layout(graph2)
    elif layout == 'kamada_kawai':
        pos1 = nx.kamada_kawai_layout(graph1)
        pos2 = nx.kamada_kawai_layout(graph2)
    else:
        raise ValueError("Unsupported layout type. Use 'spring', 'circular', or 'kamada_kawai'.")

    # Plot graph1
    node_labels1 = nx.get_node_attributes(graph1, 'label')
    edge_labels1 = nx.get_edge_attributes(graph1, 'label')
    nx.draw(graph1, pos1, with_labels=True, labels=node_labels1, ax=axes[0])
    nx.draw_networkx_edge_labels(graph1, pos1, edge_labels=edge_labels1, ax=axes[0])
    axes[0].set_title("Graph 1")

    # Plot graph2
    node_labels2 = nx.get_node_attributes(graph2, 'label')
    edge_labels2 = nx.get_edge_attributes(graph2, 'label')
    nx.draw(graph2, pos2, with_labels=True, labels=node_labels2, ax=axes[1])
    nx.draw_networkx_edge_labels(graph2, pos2, edge_labels=edge_labels2, ax=axes[1])
    axes[1].set_title("Graph 2")

    plt.show()

    return None


# print(main.calculate_cost_matrix(graphs))
# with np.printoptions(precision=4, suppress=True, floatmode = 'fixed', formatter={'float': '{:0.4f}'.format}, linewidth=100):
#     print(main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:10]}, 10, 0)[0])

cost_matrix, edit_matrix, matchings = main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:5]}, 10, 0)
# print(cost_matrix)
# print(edit_matrix)

new_graph = main.graph_matcher(graphs[2], graphs[3], edit_matrix[1,2], matchings[1,2])
print_two_graphs(graphs[3], new_graph)

