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

print(t.time() - basetime)

def BUILDNT(graph, root, height, k):
    tree = nx.Graph()
    tree.add_node(root, label=graph.nodes[root]["label"], height=0)
    D = {}
    D[root] = 0
    Phi = {}
    Phi[root] = root
    for i in range(1, height):
        F = {}
        for v in list(tree.nodes): # TODO THIS SHOULD BE LEAVES?
            for u in graph.neighbors(Phi[v]):
                if u not in D:
                    D[u] = i
                if D[u] + k >= i:
                    if u not in F:
                        c = u
                        tree.add_node(c, label=graph.nodes[c]["label"], height=i)
                        Phi[c] = u
                        F[u] = c
                    tree.add_edge(v, F[u], label=graph.edges[v, F[u]]["label"])
    return tree


def SDTED(tree1, tree2, subgraph_dict1, subgraph_dict2):

    def PAD(tree, number):
        # give the root of the tree children until it has number children
        root = list(tree.nodes)[0]
        # get the maximum node id
        new_node = max(list(tree.nodes)) + 1

        new_tree = tree.copy()

        while new_tree.degree[root] < number:
            new_tree.add_node(new_node, label = "PAD", height = tree.nodes[root]["height"] + 1)
            new_tree.add_edge(root, new_node)
            new_node += 1

        return new_tree

    def insertion_cost(tree, node, others, root):

        tree_copy = tree.copy()

        tree_copy.remove_node(root)

        for nodeX in others:
            if nodeX != node:
                tree_copy.remove_node(nodeX)

        return len(list(tree_copy.nodes)) * 2

    def deletion_cost(tree, node, others, root):

        tree_copy = tree.copy()

        tree_copy.remove_node(root)

        for nodeX in others:
            if nodeX != node:
                tree_copy.remove_node(nodeX)

        return len(list(tree_copy.nodes)) * 2


    def recusive_SDTED(tree1, tree2):
        # n is the maximum number of children of the root of the two trees
        n = max(tree1.degree[list(tree1.nodes)[0]], tree2.degree[list(tree2.nodes)[0]])

        # add undefined nodes to the trees

        tree1_padded = PAD(tree1, n)
        tree2_padded = PAD(tree2, n)

        # create the cost matrix as n x n matrix
        cost_matrix = np.zeros((n, n))

        # fill the cost matrix

        root1 = list(tree1_padded.nodes)[0]
        root2 = list(tree2_padded.nodes)[0]

        children1 = list(tree1_padded.neighbors(list(tree1_padded.nodes)[0]))
        children2 = list(tree2_padded.neighbors(list(tree2_padded.nodes)[0]))

        for i in range(n):
            for j in range(n):

                child1_ident = children1[i]
                child2_ident = children2[j]

                child1 = tree1_padded.nodes[child1_ident]
                child2 = tree2_padded.nodes[child2_ident]


                if child1["label"] != "PAD" or child2["label"] != "PAD":
                    if child1["label"] != "PAD" and child2["label"] == "PAD":
                        # cost_matrix[i][j] = insertion_cost(tree1_padded, child1_ident, children1, root1)
                        cost_matrix[i][j] = tree1_padded.nodes[child1_ident]["cost"] + 1
                    elif child2["label"] != "PAD" and child1["label"] == "PAD":
                        # cost_matrix[i][j] = deletion_cost(tree2_padded, child2_ident, children2, root2)
                        cost_matrix[i][j] = tree2_padded.nodes[child2_ident]["cost"] + 1
                    else:
                        # calculate the SDTED of the trees induced by the children
                        # create a duplicate of tree1
                        tree1_copy = tree1_padded.copy()
                        tree1_copy.remove_node(list(tree1_copy.nodes)[0])
                        # remove all children of the root that are not the i-th child
                        for child in children1:
                            if child != child1_ident:
                                tree1_copy.remove_node(child)


                        # create a duplicate of tree2
                        tree2_copy = tree2_padded.copy()
                        tree2_copy.remove_node(list(tree2_copy.nodes)[0])
                        # remove all children of the root that are not the j-th child
                        for child in children2:
                            if child != child2_ident:
                                tree2_copy.remove_node(child)


                        # add the cost to change the label of the edges
                        # get the label of the edges
                        edge1 = tree1_padded.edges[root1, child1_ident]["label"]
                        edge2 = tree2_padded.edges[root2, child2_ident]["label"]

                        if edge1 != edge2:
                            temp_cost = 1
                        else:
                            temp_cost = 0

                        if tree1_copy.nodes == {} or tree2_copy.nodes == {}:
                            cost_matrix[i][j] = (len(list(tree1_copy.nodes)) + len(list(tree2_copy.nodes))) * 2 + temp_cost
                        else:
                            cost_matrix[i][j] = recusive_SDTED(tree1_copy, tree2_copy) + temp_cost
                        # cost_matrix[i][j] = recusive_SDTED(subgraph_dict1[child1_ident], subgraph_dict2[child2_ident]) + temp_cost

        # calculate the cost of the roots
        cost_root = 1
        if tree1_padded.nodes[root1]["label"] == tree2_padded.nodes[root2]["label"]:
            cost_root = 0

        # # calculate the cost of the optimal alignment
        # cost = 0
        # for i in range(n):
        #     min_cost = float("inf")
        #     for j in range(n):
        #         if cost_matrix[i][j] < min_cost:
        #             min_cost = cost_matrix[i][j]
        #     cost += min_cost

        # use linear sum assignment to calculate the optimal alignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        cost = 0
        for i in range(n):
            cost += cost_matrix[row_ind[i]][col_ind[i]]


        return cost + cost_root


    return recusive_SDTED(tree1, tree2)

def calculate_GED(graph1, graph2, nt_dict):

    min_GED = float("inf")
    for node1 in graph1.nodes:
        for node2 in graph2.nodes:
            # how do i get the id of graph1
            nt1 = nt_dict[(graph1.graph["id"], node1)][0]
            nt2 = nt_dict[(graph2.graph["id"], node2)][0]
            diff_nodes = abs(len(nt1.nodes) - len(nt2.nodes))
            diff_edges = abs(len(nt1.edges) - len(nt2.edges))
            if diff_nodes/2 >= min_GED:
                continue
            if diff_edges >= min_GED:
                continue
            else:
                GED = SDTED(nt1, nt2, nt_dict[(graph1.graph["id"], node1)][1],nt_dict[(graph2.graph["id"], node2)][1]) #TODO: Add subgraphs here
                # print(graph1.graph["id"],graph2.graph["id"], node1, node2 ,GED)

            if GED < min_GED:
                min_GED = GED

            # print("Time to compute one GED for NT-Root Nodes " + str(node1) +" and " + str(node2) +": " + str(t.time() - basetime))


    return min_GED

def calculate_costs(tree):
    # add attribute to each node of the tree that stores the cost of inserting the node
    # the insertion cost is 1 for each node
    # when inserting a node, the node and all its children, that are in lower levels, have to be inserted

    for node in tree.nodes:
        cost = 1
        queue = deque([node])
        visited = set([node])

        while queue:
            current_node = queue.popleft()
            for neighbor in tree.neighbors(current_node):
                if neighbor not in visited and tree.nodes[neighbor]["height"] > tree.nodes[current_node]["height"]:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    cost += 2
        tree.nodes[node]["cost"] = cost

    return tree

def create_nt_dict(graphs, height, k):
    nt_dict = {}
    for i in range(len(graphs)):
        for node in graphs[i+1].nodes:
            nt = BUILDNT(graphs[i+1], node, height, k)
            nt_dict[(i+1, node)] = calculate_costs(nt), create_subgraph_dict(nt)
    return nt_dict

def create_subgraph(graph, node): #TODO: erstellt kreisfreien Subgraphen, cNTs haben aber Kreise
    subgraph = nx.Graph()
    subgraph.add_node(node, label=graph.nodes[node]["label"], height=graph.nodes[node]["height"], cost=graph.nodes[node]["cost"])
    queue = deque([node])
    visited = set([node])

    while queue:
        current_node = queue.popleft()
        for neighbor in graph.neighbors(current_node):
            if graph.nodes[neighbor]["height"] > graph.nodes[current_node]["height"]:
                subgraph.add_node(neighbor, label= graph.nodes[neighbor]["label"], height=graph.nodes[neighbor]["height"], cost=graph.nodes[neighbor]["cost"])
                subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])

                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
    return subgraph

def create_subgraph_dict(graph):
    subgraph_dict = {}
    for node in graph.nodes:
        subgraph_dict[node] = create_subgraph(graph, node)
    return subgraph_dict

def calculate_cost_matrix(graphs):
    nt_dict = create_nt_dict(graphs, 8, 0)
    cost_matrix = [["#" for i in range(len(graphs))] for j in range(len(graphs))]
    for i in range(len(graphs)):
        for j in range(i,len(graphs)):
            cost_matrix[i][j] = calculate_GED(graphs[i+1], graphs[j+1], nt_dict)
            # print("Time to compute minimal GED for Graphs " + str(i) + " and " + str(j) + ": " + str(t.time() - basetime))
            # print(cost_matrix[i][j])

    for i in range(len(graphs)):
        for j in range(i+1,len(graphs)):  # Skip the diagonal
            cost_matrix[j][i] = cost_matrix[i][j]

    print(t.time() - basetime)

    return np.matrix(cost_matrix)


# print(calculate_cost_matrix(graphs))

# print(calculate_cost_matrix({k: graphs[k] for k in list(graphs)[:10]}))

