import time as t
import random as r
import numpy as np
import main as main
import concurrent.futures
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

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
            _, _, ged, _, _ = main.calculate_GED_bgm(test_graphs[test_id], train_graphs[train_id], nt_dict, cache, height=height, k=k_param)
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

def train_kNN(graphs, n_train, n_test,height = 5, k_param = 0, method="cnt"):
    # Select random graphs for training and testing
    train_graphs_keys = r.sample(list(graphs.keys()), n_train)
    train_graphs = {k: graphs[k] for k in train_graphs_keys}

    remaining_keys = list(set(graphs.keys()) - set(train_graphs_keys))
    test_graphs_keys = r.sample(remaining_keys, n_test)
    test_graphs = {k: graphs[k] for k in test_graphs_keys}

    train_labels = np.array([graphs[k].graph["label"] for k in train_graphs_keys])
    test_labels_correct = np.array([graphs[k].graph["label"] for k in test_graphs_keys])

    # Precompute GEDs
    ged_matrix, train_ids, test_ids = precompute_geds_parallel(train_graphs, test_graphs, height=height, k_param=k_param, method=method)

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric="precomputed")

    knn.fit(ged_matrix.T, train_labels)  # Fit using GED matrix

    # Predict test labels
    test_labels_predicted = knn.predict(ged_matrix)

    correct = np.sum(test_labels_predicted == test_labels_correct)
    return (correct / n_test)
