import matplotlib.pyplot as plt
import numpy as np
import main as main

def cluster_graphs(graphs, height = 5, method = "cnt"):
    if method == "cnt":
        cost_matrix = main.calculate_cost_matrix(graphs, height)[0]
    elif method == "bgm":
        cost_matrix = main.standard_bgm_matrix(graphs)[0]
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


    for j in range(10):
        for i, graph_id in enumerate(graphss.keys()):
            # remove the graphs from its current cluster
            for label, cluster in clusters.items():
                if graph_id in cluster:
                    cluster.remove(graph_id)
                    break
            # find the smallest distance to each cluster
            distances = {label: [] for label in set(graph_labels)}
            for label,cluster in clusters.items():
                # cluster_cost = float("inf")
                # for cluster_graph_id in cluster:
                #     if cost_matrix[graph_id - 1, cluster_graph_id - 1] < cluster_cost:
                #         cluster_cost = cost_matrix[graph_id - 1, cluster_graph_id - 1]
                # distances[label] = cluster_cost

                cluster_cost = 0
                for cluster_graph_id in cluster:
                    cluster_cost += cost_matrix[graph_id - 1, cluster_graph_id - 1]
                distances[label] = cluster_cost / len(cluster)

            # add the graph to the cluster with the smallest distance
            min_index, min_distance = None, float("inf")
            for index, distance in distances.items():
                if distance < min_distance:
                    min_index = index
                    min_distance = distance
            clusters[min_index].append(graph_id)

    accuracy = 0
    for label, cluster in clusters.items():
        correct = 0
        for graph_id in cluster:
            correct += int(graphs[graph_id].graph["label"] == label)
        accuracy += correct / len(cluster)
    accuracy /= len(clusters)

    return clusters, accuracy