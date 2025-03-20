import matplotlib.pyplot as plt
import numpy as np
import main as main
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score
from scipy.stats import mode

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

def spectral_clustering(graphs, ged_matrix, n_clusters=2):
    labels = []
    for graph in graphs.values():
        labels.append(graph.graph["label"])
    labels = np.array(labels)

    sigma = np.mean(ged_matrix)  # Mean of the GED matrix
    similarity_matrix = np.exp(-ged_matrix / sigma)

    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    predicted_labels = sc.fit_predict(similarity_matrix)

    def match_labels(predicted, true):
        label_map = {}
        for cluster in np.unique(predicted):
            cluster_mask = (predicted == cluster)
            cluster_labels = true[cluster_mask]  # Die echten Labels der Cluster-Elemente
            most_common_label = mode(cluster_labels, keepdims=True).mode  # Sicher extrahieren
            
            # Falls mode() einen Skalar zurückgibt, direkt verwenden
            if isinstance(most_common_label, np.ndarray):
                most_common_label = most_common_label.item()  # In Zahl umwandeln
            
            label_map[cluster] = most_common_label
        
        return np.array([label_map[label] for label in predicted])

    mapped_labels = match_labels(predicted_labels, labels)

    # Accuracy berechnen
    accuracy = accuracy_score(labels, mapped_labels)
    return accuracy

def agglomerative_clustering(graphs, ged_matrix, n_clusters=2):
    labels = []
    for graph in graphs.values():
        labels.append(graph.graph["label"])
    labels = np.array(labels)

    sigma = np.mean(ged_matrix)  # Mean of the GED matrix
    similarity_matrix = np.exp(-ged_matrix / sigma)

    ac = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    predicted_labels = ac.fit_predict(similarity_matrix)

    def match_labels(predicted, true):
        label_map = {}
        for cluster in np.unique(predicted):
            cluster_mask = (predicted == cluster)
            cluster_labels = true[cluster_mask]  # Die echten Labels der Cluster-Elemente
            most_common_label = mode(cluster_labels, keepdims=True).mode  # Sicher extrahieren
            
            # Falls mode() einen Skalar zurückgibt, direkt verwenden
            if isinstance(most_common_label, np.ndarray):
                most_common_label = most_common_label.item()  # In Zahl umwandeln
            
            label_map[cluster] = most_common_label
        
        return np.array([label_map[label] for label in predicted])

    mapped_labels = match_labels(predicted_labels, labels)

    # Accuracy berechnen
    accuracy = accuracy_score(labels, mapped_labels)
    return accuracy

def k_metoid_clustering(graphs, ged_matrix, n_clusters=2):
    labels = []
    for graph in graphs.values():
        labels.append(graph.graph["label"])
    labels = np.array(labels)

    kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
    predicted_labels = kmedoids.fit_predict(ged_matrix)

    def match_labels(predicted, true):
        label_map = {}
        for cluster in np.unique(predicted):
            cluster_mask = (predicted == cluster)
            cluster_labels = true[cluster_mask]  # Die echten Labels der Cluster-Elemente
            most_common_label = mode(cluster_labels, keepdims=True).mode  # Sicher extrahieren
            
            # Falls mode() einen Skalar zurückgibt, direkt verwenden
            if isinstance(most_common_label, np.ndarray):
                most_common_label = most_common_label.item()  # In Zahl umwandeln
            
            label_map[cluster] = most_common_label
        
        return np.array([label_map[label] for label in predicted])

    mapped_labels = match_labels(predicted_labels, labels)

    # Accuracy berechnen
    accuracy = accuracy_score(labels, mapped_labels)
    return accuracy