import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque
import random as r
import numpy as np
from scipy.optimize import linear_sum_assignment

basetime = t.time()

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

    # Füge Knoten-Labels hinzu
    for node in graph_nodes:
        graphs[graph_id].nodes[node]["label"] = node_labels.loc[node - 1, "label"]

    # Füge Kanten-Labels hinzu
    for _, row in subgraph_edges.iterrows():
        source, target = row["source"], row["target"]
        edge_label = edge_labels.loc[edges[(edges["source"] == source) & (edges["target"] == target)].index[0], "label"]
        graphs[graph_id].edges[source, target]["label"] = edge_label
        graphs[graph_id].edges[target, source]["label"] = edge_label  # Ungerichtete Kante (symmetrisch)

print(t.time() - basetime)

# function to create an unfolding neighborhood tree (with redundancies) for a given node
def create_neighborhood_tree(graph, root):
    tree = nx.Graph()
    tree.add_node(root, label=graph.nodes[root]["label"])
    queue = deque([root])
    visited = set([root])

    while queue:
        current_node = queue.popleft()
        for neighbor in graph.neighbors(current_node):
            tree.add_node(neighbor, label= graph.nodes[neighbor]["label"])
            tree.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])

            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
    return tree

def BUILDNT(graph, root, height, k):
    tree = nx.Graph()
    tree.add_node(root, label=graph.nodes[root]["label"])
    D = {}
    D[root] = 0
    for i in range(1, height):
        F = {}
        for v in list(tree.nodes):
            for u in graph.neighbors(v):
                if u not in D:
                    D[u] = i
                if D[u] + k >= i:
                    tree.add_node(u, label=graph.nodes[u]["label"])
                    tree.add_edge(v, u)
                    F[u] = 1
    return tree

def SDTED(nt1, nt2):
    if nt1 != nt2:
        return 1
    else:
        return 0


def new_SDTED(tree1, tree2):
    
    def PAD(tree, number):
        # give the root of the tree children until it has number children
        root = list(tree.nodes)[0]
        # get the maximum node id
        new_node = max(list(tree.nodes)) + 1

        while tree.degree[root] < number:
            tree.add_node(new_node, label = "PAD")
            tree.add_edge(root, new_node)
            new_node += 1

        return tree
    
    def recusive_SDTED(tree1, tree2):
        # n is the maximum number of children of the root of the two trees
        n = max(tree1.degree[list(tree1.nodes)[0]], tree2.degree[list(tree2.nodes)[0]])
        
        # add undefined nodes to the trees

        tree1 = PAD(tree1, n)
        tree2 = PAD(tree2, n)

        # create the cost matrix as n x n matrix
        cost_matrix = np.zeros((n, n))

        # fill the cost matrix

        for i in range(n):
            for j in range(n):
                # get the i-th child of the root of tree1 and the j-th child of the root of tree2
                child1_ident = list(tree1.neighbors(list(tree1.nodes)[0]))[i]
                child2_ident = list(tree2.neighbors(list(tree2.nodes)[0]))[j]
                # those variables are only ints, i need the nodes
                child1 = tree1.nodes[child1_ident]
                child2 = tree2.nodes[child2_ident]


                if child1["label"] != "PAD" or child2["label"] != "PAD":
                    if child1["label"] != "PAD" and child2["label"] == "PAD":
                        cost_matrix[i][j] = 2
                    elif child2["label"] != "PAD" and child1["label"] == "PAD":
                        cost_matrix[i][j] = 2
                    else:
                        # calculate the SDTED of the trees induced by the children
                        # create a duplicate of tree1 
                        tree1_copy = tree1.copy()
                        tree1_copy.remove_node(list(tree1_copy.nodes)[0])
                        # remove all children of the root that are not the i-th child
                        for child in list(tree1.neighbors(list(tree1.nodes)[0])):
                            if child != child1_ident:
                                tree1_copy.remove_node(child)

                        # create a duplicate of tree2
                        tree2_copy = tree2.copy()
                        tree2_copy.remove_node(list(tree2_copy.nodes)[0])
                        # remove all children of the root that are not the j-th child
                        for child in list(tree2.neighbors(list(tree2.nodes)[0])):
                            if child != child2_ident:
                                tree2_copy.remove_node(child)

                        # add the cost to change the label of the edges
                        # get the label of the edges
                        edge1 = tree1.edges[list(tree1.nodes)[0], child1_ident]["label"]
                        edge2 = tree2.edges[list(tree2.nodes)[0], child2_ident]["label"]

                        if edge1 != edge2:
                            temp_cost = 1
                        else:
                            temp_cost = 0

                        cost_matrix[i][j] = recusive_SDTED(tree1_copy, tree2_copy) + temp_cost

        # calculate the cost of the roots
        cost_root = 1
        if tree1.nodes[list(tree1.nodes)[0]]["label"] == tree2.nodes[list(tree2.nodes)[0]]["label"]:
            cost_root = 0

        # calculate the cost of the optimal alignment
        cost = 0
        for i in range(n):
            min_cost = float("inf")
            for j in range(n):
                if cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
            cost += min_cost

        return cost + cost_root

    
    return recusive_SDTED(tree1, tree2)

def calculate_GED(graph1, graph2):

    min_GED = float("inf")
    for node1 in graph1.nodes:
        for node2 in graph2.nodes:
            nt1 = create_neighborhood_tree(graph1, node1)
            nt2 = create_neighborhood_tree(graph2, node2)
            GED = new_SDTED(nt1, nt2)

            if GED < min_GED:
                min_GED = GED

    return min_GED
            

def calculate_cost_matrix(graphs):
    cost_matrix = [["#" for i in range(len(graphs))] for j in range(len(graphs))]
    for i in range(len(graphs)):
        for j in range(i,len(graphs)):
            cost_matrix[i][j] = calculate_GED(graphs[i+1], graphs[j+1])
            print(t.time() - basetime)

    for i in range(len(graphs)):
        for j in range(i+1,len(graphs)):  # Skip the diagonal
            cost_matrix[j][i] = cost_matrix[i][j]

    return np.matrix(cost_matrix)


# print(calculate_cost_matrix(graphs))

print(calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:10]}))

# nt1 = create_neighborhood_tree(graphs[1], list(graphs[1].nodes)[0])
# nt2 = create_neighborhood_tree(graphs[2], list(graphs[2].nodes)[0])


# print(new_SDTED(nt1, nt1))
# print(new_SDTED(nt1, nt2))



