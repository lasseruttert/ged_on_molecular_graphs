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
    # select first n_train graphs as training set
    train_graphs = {i: graphs[i] for i in range(1, n_train + 1)}
    # select next n_test graphs as test set
    test_graphs = {i: graphs[i] for i in range(n_train + 1, n_train + n_test + 1)}
    # create labels
    train_labels = [graph.graph["label"] for graph in train_graphs.values()]
    test_labels_correct = [graph.graph["label"] for graph in test_graphs.values()]


    # Precompute GEDs
    ged_matrix, train_ids, test_ids = precompute_geds_parallel(train_graphs, test_graphs, height=height, k_param=k_param, method=method)

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric="precomputed")

    knn.fit(ged_matrix.T, train_labels)  # Fit using GED matrix

    # Predict test labels
    test_labels_predicted = knn.predict(ged_matrix)

    correct = np.sum(test_labels_predicted == test_labels_correct)
    return (correct / n_test)

def knn_graph_classification(query_graph, train_graphs, method="cnt", height=5, k_param=0):
    distances = []
    cache = {}
    for train_graph in train_graphs:
        if method == "cnt":
            _, _, ged, _, _ = main.calculate_GED_bgm(graph1=query_graph, graph2=train_graph, cache=cache, height=height, k=k_param)
        if method == "bgm":
            _, _, ged, _, _ = main.standard_bgm(query_graph, train_graph)
        distances.append(ged)
    
    nearest_neighbor_index = np.argmin(distances)
    return train_graphs[nearest_neighbor_index].graph["label"]

def test_knn(graphs, method="cnt", height=5, k_param=0):
    n = len(graphs)
    n_train = int(0.75 * n)  # 80% für Training
    n_test = n - n_train    # 20% für Test
    graph_list = list(graphs.values())

    train_graphs = graph_list[:n_train]
    test_graphs = graph_list[n_train:]

    tp, fp, tn, fn = 0, 0 ,0 ,0
    for i in range(1, len(test_graphs)):
        classification = knn_graph_classification(test_graphs[i], train_graphs, method=method, height=height, k_param=k_param)
        test_label = test_graphs[i].graph["label"]
        if classification == 1:
            if test_label == 1:
                tp += 1
            else:
                fp += 1
        elif classification == -1 or classification == 0:
            if test_label == -1 or test_label == 0:
                tn += 1
            else:
                fn += 1

    return tp, fp, tn, fn
