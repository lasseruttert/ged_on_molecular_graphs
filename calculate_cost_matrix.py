import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque
import random as r
import numpy as np
from scipy.optimize import linear_sum_assignment
import main as main
import os

basetime = t.time()

# Pfad zu den Dateien
current_dir = os.path.dirname(__file__)
dataset_name = "PTC_FM"
path = os.path.join(current_dir, "data", dataset_name)

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
if os.path.exists(f"{path}\{dataset_name}_node_labels.txt"):
    node_labels = pd.read_csv(f"{path}\{dataset_name}_node_labels.txt", header=None)
    node_labels.columns = ["label"]

# Lade Kanten-Labels
if os.path.exists(f"{path}\{dataset_name}_edge_labels.txt"):
    edge_labels = pd.read_csv(f"{path}\{dataset_name}_edge_labels.txt", header=None)
    edge_labels.columns = ["label"]

# Erstelle die Graphen
graphs = {}
for graph_id in node_to_graph["graph_id"].unique():
    graph_nodes = node_to_graph[node_to_graph["graph_id"] == graph_id].index + 1
    subgraph_edges = edges[edges["source"].isin(graph_nodes) & edges["target"].isin(graph_nodes)]
    graphs[graph_id] = nx.from_pandas_edgelist(subgraph_edges, source="source", target="target")
    graphs[graph_id].add_nodes_from(graph_nodes)
    graphs[graph_id].graph["label"] = graph_labels.loc[graph_id - 1, "label"]
    # gib jeden Graphen eine ID
    graphs[graph_id].graph["id"] = graph_id

    # Füge Knoten-Labels hinzu
    if "node_labels" in locals():
        for node in graph_nodes:
            graphs[graph_id].nodes[node]["label"] = node_labels.loc[node - 1, "label"]
    else: 
        for node in graph_nodes:
            graphs[graph_id].nodes[node]["label"] = "dummy"

    # Füge Kanten-Labels hinzu
    if "edge_labels" in locals():
        for _, row in subgraph_edges.iterrows():
            source, target = row["source"], row["target"]
            edge_label = edge_labels.loc[edges[(edges["source"] == source) & (edges["target"] == target)].index[0], "label"]
            graphs[graph_id].edges[source, target]["label"] = edge_label
            graphs[graph_id].edges[target, source]["label"] = edge_label  # Ungerichtete Kante (symmetrisch)
    else:
        for _, row in subgraph_edges.iterrows():
            source, target = row["source"], row["target"]
            graphs[graph_id].edges[source, target]["label"] = "dummy"
            graphs[graph_id].edges[target, source]["label"] = "dummy"

print(f"Loading the Graphs: {t.time() - basetime}s")

import seaborn as sns

# Assuming cost_matrix is your matrix
def plot_cost_matrix(cost_matrix, title="Cost Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cost_matrix, annot=True, fmt="d", cmap="coolwarm", cbar=True)
    plt.xlabel('Graph 1 Nodes')
    plt.ylabel('Graph 2 Nodes')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    n = 20

    cost_matrix, edit_matrix, matchings = main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:n]}, 10, 0)

    plot_cost_matrix(cost_matrix, title="Cost Matrix")

    # avg value of cost_matrix
    print(np.mean(cost_matrix))

    # highest difference in cost_matrix
    # find min, not on diagonal
    print(np.max(cost_matrix) - np.min(cost_matrix[np.nonzero(cost_matrix)]))

    for i in range(n): 
        for j in range(n):
            if i == j:
                continue
            if i > j:
                continue
            new_graph = main.graph_matcher(graphs[i+1], graphs[j+1], edit_matrix[i,j], matchings[i,j])
            current_bool = main.isomorph_check(graphs[j+1], new_graph)
            if not current_bool:
                print(edit_matrix[i,j])
                print(matchings[i,j])
                main.print_two_graphs(graphs[j+1], new_graph)
                print(i+1, j+1)
                print("\n")
                print("\n")

    bgm_cost_matrix, bgm_edit_matrix, bgm_matchings = main.standard_bgm_matrix({k: graphs[k] for k in list(graphs)[:n]})

    print(bgm_cost_matrix)

    # avg value of cost_matrix
    print(np.mean(bgm_cost_matrix))

    # highest difference in cost_matrix
    # find min, not on diagonal
    print(np.max(bgm_cost_matrix) - np.min(bgm_cost_matrix[np.nonzero(bgm_cost_matrix)]))

    for i in range(n): 
        for j in range(n):
            if i == j:
                continue
            if i > j:
                continue
            new_graph = main.graph_matcher(graphs[i+1], graphs[j+1], bgm_edit_matrix[i,j], bgm_matchings[i,j])
            current_bool = main.isomorph_check(graphs[j+1], new_graph)
            if not current_bool:
                print(bgm_edit_matrix[i,j])
                print(bgm_matchings[i,j])
                main.print_two_graphs(graphs[j+1], new_graph)
                print(i+1, j+1)
                print("\n")
                print("\n")

    # save cost matrix to file
    # np.savetxt(f"{dataset_name}_cost_matrix.csv", cost_matrix, delimiter=",")

    print(cost_matrix - bgm_cost_matrix)

    print("Done")
