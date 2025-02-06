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
dataset_name = "MUTAG"
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

def cluster_graphs(graphs, cost_matrix):
    graphss = graphs.copy()
    # get all graph labels
    graph_labels = [graphss[key].graph["label"] for key in graphss.keys()]

    # create a cluster for every label
    clusters = {label: [] for label in set(graph_labels)}

    to_delete = []
    for label in set(graph_labels):
        for graph_id in list(graphss.keys()):  # `list()` macht eine Kopie der Keys
            if graphss[graph_id].graph["label"] == label:
                clusters[label].append(graph_id)
                to_delete.append(graph_id)  # Merke den zu löschenden Graphen
                break
    for graph_id in to_delete:
        del graphss[graph_id]



    for i, graph_id in enumerate(graphss.keys()):
        # find the smallest distance to each cluster
        distances = []
        for cluster in clusters.values():
            cluster_cost = 0
            for cluster_graph_id in cluster:
                cluster_cost += cost_matrix[graph_id - 1, cluster_graph_id - 1]
            distances.append(cluster_cost/len(cluster))
        # add the graph to the cluster with the smallest distance
        label = graphs[graph_id].graph["label"]  # Hole das Label des Graphen
        clusters[label].append(graph_id)


    return clusters

if __name__ == "__main__":
    n = 20

    print("CNT")
    used_graphs = {key: graphs[key] for key in list(graphs.keys())[:n]}
    cost_matrixs = main.calculate_cost_matrix(used_graphs)[0]
    print(np.mean(cost_matrixs))  # Sollte <class 'numpy.ndarray'> sein, nicht <class 'list'>

    clusters = cluster_graphs(used_graphs, cost_matrixs)
    
    for cluster_label, cluster_graphss in clusters.items():
        print(f"Cluster {cluster_label}")
        print(cluster_graphss)
        correct = 0
        for graph_id in cluster_graphss:
            correct += int(graphs[graph_id].graph["label"] == cluster_label)
        print(f"Correct: {correct}/{len(cluster_graphss)}")

    print("BGM")
    used_graphs = {key: graphs[key] for key in list(graphs.keys())[:n]}    
    cost_matrixx = main.standard_bgm_matrix(used_graphs)[0]
    print(np.mean(cost_matrixx))  # Sollte <class 'numpy.ndarray'> sein, nicht <class 'list'>

    clusters = cluster_graphs(used_graphs, cost_matrixx)

    for cluster_label, cluster_graphs in clusters.items():
        print(f"Cluster {cluster_label}")
        print(cluster_graphs)
        correct = 0
        for graph_id in cluster_graphs:
            correct += int(graphs[graph_id].graph["label"] == cluster_label)
        print(f"Correct: {correct}/{len(cluster_graphs)}")

    print("Done!")