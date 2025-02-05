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


if __name__ == "__main__":
    # total_time = 0
    # for i in range(10):
    #     basetime = t.time()
    #     main.calculate_GED_bgm(graphs[1], graphs[2])
    #     total_time += t.time() - basetime

    # print(f"Average time: {total_time / 10}s")

    # heights = [1,2,3,4,5,6,7,8,9,10]
    # runtimes = []
    # for height in heights:
    #     total_time = 0
    #     for i in range(100):
    #         x = r.randint(1, 188)
    #         y = r.randint(1, 188)
    #         basetime = t.time()
    #         main.calculate_GED_bgm(graphs[x], graphs[y], height=height)
    #         total_time += t.time() - basetime
    #     runtimes.append(total_time / 100)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, runtimes)
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title(f"Runtime of GED calculation: {dataset_name}")

    # plt.savefig(f"runtime_ged_{dataset_name}.png") 

    # heights = [1,2,3,4,5,6,7,8,9,10]
    # runtimes = []
    # for height in heights:
    #     total_time = 0
    #     for i in range(10):
    #         basetime = t.time()
    #         main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:5]}, height=height, k=0)
    #         total_time += t.time() - basetime
    #     runtimes.append(total_time / 10)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, runtimes)
    # plt.xlabel("Height")
    # plt.ylabel("Runtime")
    # plt.title(f"Runtime of cost matrix calculation: {dataset_name}")

    # plt.savefig(f"runtime_cost_matrix_{dataset_name}.png")

    # avg error
    # heights = [1,2,3,4,5,6,7,8,9,10]
    # error = []
    # actual = 10
    # for height in heights:
    #     print(f"Height: {height}")
    #     current_error = 0
    #     for i in range(10):
    #         _,_,calculated,_,_ = main.calculate_GED_bgm(graphs[2], graphs[3], height=height)
    #         current_error += (abs(calculated - actual)/actual)
    #     error.append(current_error / 10)

    # # save a plot of the runtime, y axis is the runtime, x axis is height parameter
    # plt.plot(heights, error)
    # plt.xlabel("Height")
    # plt.ylabel("Error")
    # plt.title(f"Error of GED calculation: {dataset_name}")

    # plt.savefig(f"error_ged_{dataset_name}.png")

    print("Done")