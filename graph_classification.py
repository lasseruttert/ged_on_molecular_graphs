import time as t
import random as r
import numpy as np
import main as main
import concurrent.futures
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

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


def precompute_geds_parallel(train_graphs, test_graphs, height=8, k_param=0, method="cnt"):
    """Parallelized GED precomputation using ProcessPoolExecutor."""
    
    all_graphs = train_graphs.copy()
    for test_graph in test_graphs.values():
        all_graphs[test_graph.graph["id"]] = test_graph.copy()
    
    if method == "cnt":
        nt_dict = main.create_nt_dict(all_graphs, height, k_param)
    
    train_ids = list(train_graphs.keys())
    test_ids = list(test_graphs.keys())
    ged_matrix = np.zeros((len(test_ids), len(train_ids)))

    cache = {}


    def compute_ged(i, j, test_id, train_id, cache=cache):
        if method == "cnt":
            _, _, ged, _, _ = main.calculate_GED_bgm(test_graphs[test_id], train_graphs[train_id], nt_dict, cache,5)
        if method == "bgm":
            _, _, ged, _, _ = main.standard_bgm(test_graphs[test_id], train_graphs[train_id])
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

    graphs = main.load_graphs("MUTAG")

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


    # Precompute GEDs
    ged_matrix, train_ids, test_ids = precompute_geds_parallel(train_graphs, test_graphs, method="cnt")

    # Train k-NN classifier
    knn1 = KNeighborsClassifier(n_neighbors=3, metric="precomputed")

    knn1.fit(ged_matrix.T, train_labels)  # Fit using GED matrix

    # Predict test labels
    test_labels_predicted = knn1.predict(ged_matrix)

    correct = np.sum(test_labels_predicted == test_labels_correct)
    print(f"CNT - Accuracy: {correct / n_test}")

    # Precompute GEDs using BGM
    ged_matrix_bgm, train_ids, test_ids = precompute_geds_parallel(train_graphs, test_graphs, method="bgm")

    # Train k-NN classifier
    knn2 = KNeighborsClassifier(n_neighbors=3, metric="precomputed")

    knn2.fit(ged_matrix_bgm.T, train_labels)  # Fit using GED matrix

    # Predict test labels
    test_labels_predicted_bgm = knn2.predict(ged_matrix_bgm)

    correct_bgm = np.sum(test_labels_predicted_bgm == test_labels_correct)
    print(f"BGM - Accuracy: {correct_bgm / n_test}")

    print("Done")