import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque
import random as r
import numpy as np
from scipy.optimize import linear_sum_assignment
import main as main

# Pfad zu den Dateien
path = "G:\Meine Ablage\PG\ged_on_molecular_graphs\data\MUTAG"
dataset_name = "MUTAG"

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

# print(main.calculate_cost_matrix(graphs))
# with np.printoptions(precision=4, suppress=True, floatmode = 'fixed', formatter={'float': '{:0.4f}'.format}, linewidth=100):
#     print(main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:5]}))

df = pd.DataFrame(main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:10]}))
print(df.to_string(index=False, header=False, float_format=lambda x: f"{int(x)}" if x == int(x) else f"{x:.2f}"))

# import cProfile
# import pstats

# def profile_code():
#     profiler = cProfile.Profile()
#     profiler.enable()

#     # Call the function you want to profile
#     main.calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:10]})

#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

# # Call the profiling function
# profile_code()

# nt_dic = main.create_nt_dict({k: graphs[k] for k in list(graphs)[:2]}, 8, 0)

# node1_of_tree1 = list(graphs[1].nodes)[0]
# node1_of_tree2 = list(graphs[2].nodes)[0]

# main.SDTED(nt_dic[(1, node1_of_tree1)][0], nt_dic[(2, node1_of_tree2)][0], nt_dic[(1, node1_of_tree1)][1], nt_dic[(2, node1_of_tree2)][1])
