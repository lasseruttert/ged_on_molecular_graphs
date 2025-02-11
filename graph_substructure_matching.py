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
from itertools import combinations

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

def create_subgraph(graph, nodes):
    subgraph = nx.Graph()
    subgraph.graph["id"] = graph.graph["id"]
    for node in nodes:
        subgraph.add_node(node, label=graph.nodes[node]["label"])
    for source, target in graph.edges:
        if source in nodes and target in nodes:
            subgraph.add_edge(source, target, label=graph.edges[source, target]["label"])
    return subgraph

def create_all_subgraphs(graph):
    all_subgraphs = []
    for i in range(1, len(graph.nodes) + 1):
        for subgraph_nodes in combinations(graph.nodes, i):
            all_subgraphs.append(create_subgraph(graph, subgraph_nodes))
    return all_subgraphs

def create_benzol_ring():
    ring = nx.Graph()
    ring.graph["id"] = "benzol"
    ring.add_node(1, label="0")
    ring.add_node(2, label="0")
    ring.add_node(3, label="0")
    ring.add_node(4, label="0")
    ring.add_node(5, label="0")
    ring.add_node(6, label="0")

    ring.add_edge(1, 2, label="0")
    ring.add_edge(2, 3, label="0")
    ring.add_edge(3, 4, label="0")
    ring.add_edge(4, 5, label="0")
    ring.add_edge(5, 6, label="0")
    ring.add_edge(6, 1, label="0")
    return ring

if __name__ == "__main__":
    # matched_substructures = {}
    # subgraphs = create_all_subgraphs(graphs[1])

    ring = create_benzol_ring()

    # i = 0

    # for subgraph in subgraphs:
    #     # check if subgraph is connected
    #     if nx.is_connected(subgraph):
    #         _,_,GED,_,_ = main.calculate_GED_bgm(subgraph, ring)
    #         matched_substructures[i] = (GED,subgraph)
    #         i += 1

    # sorted_matches = dict(sorted(matched_substructures.items(), key=lambda item: item[1][0]))
    # min_key, (min_ged, min_subgraph) = min(matched_substructures.items(), key=lambda item: item[1][0])

    # main.print_two_graphs(min_subgraph, ring)
    nodes1 = [1,2,3,4,5,6]
    nodes2 = [4,5,7,8,9,10]
    nodes3 = [9,10,11,12,13,14]

    subgraph1 = create_subgraph(graphs[1],nodes1)
    subgraph2 = create_subgraph(graphs[1],nodes2)
    subgraph3 = create_subgraph(graphs[1],nodes3)

    # main.print_two_graphs(subgraph1,ring)
    # main.print_two_graphs(subgraph2,ring)
    # main.print_two_graphs(subgraph3,ring)
    
    _,_,ged1,e1,m1 = main.calculate_GED_bgm(subgraph1, ring)
    _,_,ged2,e2,m2 = main.calculate_GED_bgm(subgraph2, ring)
    _,_,ged3,e3,m3 = main.calculate_GED_bgm(subgraph3, ring)

    g1 = main.graph_matcher(subgraph1,ring,e1,m1)
    g2 = main.graph_matcher(subgraph2,ring,e2,m2)
    g3 = main.graph_matcher(subgraph3,ring,e3,m3)

    print(main.isomorph_check(g1,ring))
    print(main.isomorph_check(g2,ring))
    print(main.isomorph_check(g3,ring))

    print(ged1)
    print(ged2)
    print(ged3)

    # _,_,ged_bgm1,_,_ = main.standard_bgm(subgraph1, ring)
    # _,_,ged_bgm2,_,_ = main.standard_bgm(subgraph2, ring)
    # _,_,ged_bgm3,_,_ = main.standard_bgm(subgraph3, ring)

    # print(ged_bgm1)
    # print(ged_bgm2)
    # print(ged_bgm3)

# ! GED cannot be used as a similarity measure for subgraph matching

    print("Done!")