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

from collections import Counter

def classify_graph_kNN(test_graph, train_graphs, train_labels, k=3, height=8, k_param=0, cache= {}):
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

    all_graphs = train_graphs.copy()  # Now, all_graphs is a separate dictionary
    all_graphs[test_graph.graph["id"]] = test_graph


    nt_dict = main.create_nt_dict(all_graphs, height, k_param)
    ged_distances = {}

    for graph_id, train_graph in train_graphs.items():
        _, _, ged, _, _ = main.calculate_GED_bgm(test_graph, train_graph, nt_dict, cache)
        ged_distances[graph_id] = ged

    # Sortiere nach GED-Distanzen (kleinste zuerst)
    # Ensure that nearest neighbors are only from the training set
    nearest_neighbors = [g for g in sorted(ged_distances, key=ged_distances.get) if g in train_labels][:k]


    # Mehrheit der k-nächsten Labels bestimmen
    neighbor_labels = [train_labels[n] for n in nearest_neighbors]
    predicted_label = Counter(neighbor_labels).most_common(1)[0][0]

    return predicted_label

# Precompute GED for all train-test pairs
def precompute_geds(test_graphs, train_graphs, height=8, k_param=0):
    nt_dict = main.create_nt_dict({**train_graphs, **test_graphs}, height, k_param)
    ged_distances = {}
    for test_id, test_graph in test_graphs.items():
        ged_distances[test_id] = {}
        for train_id, train_graph in train_graphs.items():
            _, _, ged, _, _ = main.calculate_GED_bgm(test_graph, train_graph, nt_dict, {})
            ged_distances[test_id][train_id] = ged
    return ged_distances

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random as r
import concurrent.futures

def precompute_geds_parallel(train_graphs, test_graphs, height=8, k_param=0):
    """Parallelized GED precomputation using ProcessPoolExecutor."""
    
    all_graphs = train_graphs.copy()
    for test_graph in test_graphs.values():
        all_graphs[test_graph.graph["id"]] = test_graph.copy()
    
    nt_dict = main.create_nt_dict(all_graphs, height, k_param)
    
    train_ids = list(train_graphs.keys())
    test_ids = list(test_graphs.keys())
    ged_matrix = np.zeros((len(test_ids), len(train_ids)))

    cache = {}


    def compute_ged(i, j, test_id, train_id, cache=cache):
        _, _, ged, _, _ = main.calculate_GED_bgm(test_graphs[test_id], train_graphs[train_id], nt_dict, cache)
        return i, j, ged

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(compute_ged, i, j, test_ids[i], train_ids[j], cache)
            for i in range(len(test_ids))
            for j in range(len(train_ids))
        }
        for future in concurrent.futures.as_completed(futures):
            i, j, ged = future.result()
            ged_matrix[i, j] = ged

    return ged_matrix, train_ids, test_ids


if __name__ == "__main__":
    # Select random graphs for training and testing
    n_train = 90
    n_test = 90
    train_graphs_keys = r.sample(list(graphs.keys()), n_train)
    train_graphs = {k: graphs[k] for k in train_graphs_keys}

    remaining_keys = list(set(graphs.keys()) - set(train_graphs_keys))
    test_graphs_keys = r.sample(remaining_keys, n_test)
    test_graphs = {k: graphs[k] for k in test_graphs_keys}

    train_labels = np.array([graphs[k].graph["label"] for k in train_graphs_keys])
    test_labels_correct = np.array([graphs[k].graph["label"] for k in test_graphs_keys])

    # Precompute GED matrix
    basetime = t.time()
    ged_matrix, train_ids, test_ids = precompute_geds_parallel(train_graphs, test_graphs)
    print(f"GED Matrix: {t.time() - basetime}s")

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric="precomputed")



    knn.fit(ged_matrix.T, train_labels)  # Fit using GED matrix

    # Predict test labels
    test_labels_predicted = knn.predict(ged_matrix)

    correct = np.sum(test_labels_predicted == test_labels_correct)
    print(f"Accuracy: {correct / n_test}")

    print("Done")