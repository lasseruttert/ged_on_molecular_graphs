import matplotlib.pyplot as plt
import numpy as np
import main as main

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
        distances = {label: [] for label in set(graph_labels)}
        for label,cluster in clusters.items():
            cluster_cost = float("inf")
            for cluster_graph_id in cluster:
                if cost_matrix[graph_id - 1, cluster_graph_id - 1] < cluster_cost:
                    cluster_cost = cost_matrix[graph_id - 1, cluster_graph_id - 1]
            distances[label] = cluster_cost
        # add the graph to the cluster with the smallest distance
        min_index, min_distance = None, float("inf")
        for index, distance in distances.items():
            if distance < min_distance:
                min_index = index
                min_distance = distance
        clusters[min_index].append(graph_id)


    return clusters

if __name__ == "__main__":
    n = 50
    graphs = main.load_graphs("MUTAG",n)

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