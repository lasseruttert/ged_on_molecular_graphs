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

print(f"Loading the Graphs: {t.time() - basetime}s")

from collections import Counter

def classify_graph_kNN(test_graph, train_graphs, train_labels, k=3, height=8, k_param=0):
    """
    Klassifiziert einen Test-Graphen basierend auf k-NN und GED.
    
    Parameters:
        test_graph (nx.Graph): Der zu klassifizierende Graph.
        train_graphs (dict): Trainingsgraphen {graph_id: nx.Graph}
        train_labels (dict): Labels {graph_id: label}
        k (int): Anzahl der nächsten Nachbarn.
        height (int): Parameter für GED.
        k_param (int): Parameter für GED.
    
    Returns:
        predicted_label: Die vorhergesagte Klasse für test_graph.
    """
    nt_dict = main.create_nt_dict(train_graphs, height, k_param)
    ged_distances = {}

    for graph_id, train_graph in train_graphs.items():
        _, _, ged, _, _ = main.calculate_GED_bgm(test_graph, train_graph, nt_dict, {})
        ged_distances[graph_id] = ged

    # Sortiere nach GED-Distanzen (kleinste zuerst)
    nearest_neighbors = sorted(ged_distances, key=ged_distances.get)[:k]

    # Mehrheit der k-nächsten Labels bestimmen
    neighbor_labels = [train_labels[n] for n in nearest_neighbors]
    predicted_label = Counter(neighbor_labels).most_common(1)[0][0]

    return predicted_label

from sklearn.svm import SVC

def train_svm_with_ged(train_graphs, train_labels, height=8, k_param=0):
    """
    Trainiert eine SVM mit GED als Kernel-Matrix.
    
    Parameters:
        train_graphs (dict): Trainingsgraphen {graph_id: nx.Graph}
        train_labels (dict): Labels {graph_id: label}
        height (int): Parameter für GED.
        k_param (int): Parameter für GED.
    
    Returns:
        svm_model: Trainiertes SVM-Modell.
    """
    graph_ids = list(train_graphs.keys())
    nt_dict = main.create_nt_dict(train_graphs, height, k_param)

    # GED-Matrix berechnen
    num_graphs = len(graph_ids)
    ged_matrix = np.zeros((num_graphs, num_graphs))

    for i, g1 in enumerate(graph_ids):
        for j, g2 in enumerate(graph_ids):
            if i <= j:  # Berechne nur obere Hälfte der Matrix
                _, _, ged, _, _ = main.calculate_GED_bgm(train_graphs[g1], train_graphs[g2], nt_dict, {})
                ged_matrix[i, j] = ged
                ged_matrix[j, i] = ged  # Symmetrisch

    # SVM mit vorgegebener Distanzmatrix als Kernel
    svm_model = SVC(kernel="precomputed")
    svm_model.fit(ged_matrix, [train_labels[g] for g in graph_ids])

    return svm_model

def predict_svm_with_ged(svm_model, test_graph, train_graphs, height=8, k_param=0):
    """
    Macht Vorhersagen für einen Test-Graphen mit einer SVM und GED.
    
    Parameters:
        svm_model: Das trainierte SVM-Modell.
        test_graph (nx.Graph): Der zu klassifizierende Graph.
        train_graphs (dict): Trainingsgraphen {graph_id: nx.Graph}
        height (int): Parameter für GED.
        k_param (int): Parameter für GED.
    
    Returns:
        predicted_label: Die vorhergesagte Klasse für test_graph.
    """
    nt_dict = main.create_nt_dict(train_graphs, height, k_param)
    graph_ids = list(train_graphs.keys())

    # GED zwischen Test-Graph und Trainingsgraphen berechnen
    ged_vector = np.zeros((1, len(graph_ids)))
    for i, g in enumerate(graph_ids):
        _, _, ged, _, _ = main.calculate_GED_bgm(test_graph, train_graphs[g], nt_dict, {})
        ged_vector[0, i] = ged

    return svm_model.predict(ged_vector)[0]


if __name__ == "__main__":
    n = 10

    cost_matrix, edit_matrix, matchings = main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:n]}, 10, 0)

    print(cost_matrix)

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

    print("Done")

    # TODO: more experiments
    # * Graph Classification
    # * Graph Clustering
    # * Outlier Detection
    # * compare with other algorithms
