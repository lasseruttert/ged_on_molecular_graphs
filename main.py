import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as t
from collections import deque
import random as r
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations
from functools import lru_cache

def BUILDNT(graph, root, height, k):
    tree = nx.DiGraph()
    tree.add_node(root, label=graph.nodes[root]["label"], height=0)
    D = {}
    D[root] = 0
    Phi = {}
    Phi[root] = root
    for i in range(1, height):
        F = {}
        for v in list(tree.nodes):
            if tree.out_degree(v) == 0:
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

    tree.graph["encoding"] = frozenset(tree.nodes)

    return tree


def SDTED(treee1, treee2, subgraph_dict1, subgraph_dict2, cache):
    def PAD(tree, number):
        root = next(iter(tree.nodes))
        
        current_degree = tree.degree[root]
        if current_degree >= number:
            return tree

        missing_children = number - current_degree
        
        new_node = max(tree.nodes) + 1
        
        # Knoten und Kanten effizient hinzufügen
        new_nodes = [(new_node + i, {"label": "PAD", "height": tree.nodes[root]["height"] + 1}) for i in range(missing_children)]
        new_edges = [(root, new_node + i) for i in range(missing_children)]
        
        tree.add_nodes_from(new_nodes)
        tree.add_edges_from(new_edges)

        return tree

    @lru_cache(maxsize=None)
    def recusive_SDTED(tree1, tree2):
        key = (tree1.graph["encoding"], tree2.graph["encoding"])
        if key in cache:
            return cache[key]

        # n is the maximum number of children of the root of the two trees
        n = max(tree1.degree[next(iter(tree1.nodes))], tree2.degree[next(iter(tree2.nodes))])

        # add undefined nodes to the trees

        tree1_padded = tree1
        tree2_padded = tree2

        if tree1.degree[next(iter(tree1.nodes))] != tree2.degree[next(iter(tree2.nodes))]:

            tree1_padded = PAD(tree1, n)
            tree2_padded = PAD(tree2, n)

        # create the cost matrix as n x n matrix
        cost_matrix = np.zeros((n, n))

        # fill the cost matrix

        root1 = next(iter(tree1_padded.nodes))
        root2 = next(iter(tree2_padded.nodes))

        children1 = list(tree1_padded.neighbors(next(iter(tree1_padded.nodes))))
        children2 = list(tree2_padded.neighbors(next(iter(tree2_padded.nodes))))

        for i in range(n):
            for j in range(n):

                child1_ident = children1[i]
                child2_ident = children2[j]

                child1 = tree1_padded.nodes[child1_ident]
                child2 = tree2_padded.nodes[child2_ident]

                child1_label = child1["label"]
                child2_label = child2["label"]

                if child1_label != "PAD" or child2_label != "PAD":
                    if child1_label != "PAD" and child2_label == "PAD":
                        cost_matrix[i][j] = tree1_padded.nodes[child1_ident]["cost"] + 1
                    elif child2_label != "PAD" and child1_label == "PAD":
                        cost_matrix[i][j] = tree2_padded.nodes[child2_ident]["cost"] + 1
                    else:
                        temp_cost = 1 if hash(tree1_padded.edges[root1, child1_ident]["label"]) != hash(tree2_padded.edges[root2, child2_ident]["label"]) else 0
                        cost_matrix[i][j] = recusive_SDTED(subgraph_dict1[child1_ident], subgraph_dict2[child2_ident]) + temp_cost

        # calculate the cost of the roots
        cost_root = 1
        if tree1_padded.nodes[root1]["label"] == tree2_padded.nodes[root2]["label"]:
            cost_root = 0

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        cost = 0
        for i in range(n): 
            cost += cost_matrix[row_ind[i]][col_ind[i]]

        cache[key] = cost + cost_root

        return cost + cost_root

    result = recusive_SDTED(treee1, treee2)
    cache[(treee1.graph["encoding"], treee2.graph["encoding"])] = result
    return result


def calculate_costs(tree):
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


def create_subgraph(graph, node):
    subgraph = nx.DiGraph()
    subgraph.add_node(node, label=graph.nodes[node]["label"], height=graph.nodes[node]["height"], cost=graph.nodes[node]["cost"])

    current_level = [node]
    next_level = []

    while current_level:
        for current_node in current_level:
            for neighbor in graph.neighbors(current_node):
                if graph.nodes[neighbor]["height"] > graph.nodes[current_node]["height"]:
                    if neighbor not in subgraph.nodes:
                        subgraph.add_node(neighbor, label=graph.nodes[neighbor]["label"], height=graph.nodes[neighbor]["height"], cost=graph.nodes[neighbor]["cost"])
                        subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])
                        next_level.append(neighbor)
                    else:
                        subgraph.add_edge(current_node, neighbor, label=graph.edges[current_node, neighbor]["label"])
                    

        current_level = next_level
        next_level = []

    subgraph.graph["encoding"] = frozenset(subgraph.nodes)

    return subgraph


def create_subgraph_dict(graph):
    subgraph_dict = {}
    for node in graph.nodes:
        subgraph_dict[node] = create_subgraph(graph, node)
    return subgraph_dict


def create_nt_dict(graphs, height, k):
    nt_dict = {}
    for graph_id in graphs:
        for node in graphs[graph_id].nodes:
            nt = BUILDNT(graphs[graph_id], node, height, k)
            nt_dict[(graph_id, node)] = calculate_costs(nt), create_subgraph_dict(nt)
    return nt_dict


# def calculate_GED(graph1, graph2, nt_dict):
#     min_GED = float("inf")
#     for node1 in graph1.nodes:
#         for node2 in graph2.nodes:
#             nt1 = nt_dict[(graph1.graph["id"], node1)][0]
#             nt2 = nt_dict[(graph2.graph["id"], node2)][0]
#             # diff_nodes = abs(len(nt1.nodes) - len(nt2.nodes))
#             # diff_edges = abs(len(nt1.edges) - len(nt2.edges))
#             # if diff_nodes/2 >= min_GED:
#             #     continue
#             # if diff_edges >= min_GED:
#             #     continue
#             # else:
#             GED = SDTED(nt1, nt2, nt_dict[(graph1.graph["id"], node1)][1],nt_dict[(graph2.graph["id"], node2)][1]) #TODO: Add subgraphs here
#             print(graph1.graph["id"],graph2.graph["id"], node1, node2 ,GED)

#             if GED < min_GED:
#                 min_GED = GED

#             # print("Time to compute one GED for NT-Root Nodes " + str(node1) +" and " + str(node2) +": " + str(t.time() - basetime))

#     return min_GED

# def calculate_cost_matrix(graphs):
#     basetime = t.time()
#     nt_dict = create_nt_dict(graphs, 8, 0)

#     graph_ids = list(graphs.keys())
#     cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)

#     for i, j in combinations(range(len(graph_ids)), 2):
#         cost_matrix[i, j] = cost_matrix[j, i] = calculate_GED(graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict)

#     print(f"Total time: {t.time() - basetime}")
#     return np.matrix(cost_matrix)


import concurrent.futures
from networkx.algorithms import isomorphism

def calculate_GED_parallel(graph1, graph2, nt_dict, cache):
    min_GED = float("inf")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for node1 in graph1.nodes:
            for node2 in graph2.nodes:
                nt1 = nt_dict[(graph1.graph["id"], node1)][0]
                nt2 = nt_dict[(graph2.graph["id"], node2)][0]
                nt1_subgraph = nt_dict[(graph1.graph["id"], node1)][1]
                nt2_subgraph = nt_dict[(graph2.graph["id"], node2)][1]

                diff_nodes = abs(len(nt1.nodes) - len(nt2.nodes))
                diff_edges = abs(len(nt1.edges) - len(nt2.edges))

                if diff_nodes/2 >= min_GED:
                    continue
                if diff_edges >= min_GED:
                    continue

                futures.append(executor.submit(SDTED, nt1, nt2, nt1_subgraph, nt2_subgraph, cache))

        for future in concurrent.futures.as_completed(futures):
            GED = future.result()
            if GED < min_GED:
                min_GED = GED

    return min_GED

def calculate_cost_matrix(graphs, height=8, k=0):
    basetime = t.time()
    cache = {}
    nt_dict = create_nt_dict(graphs, height, k)

    graph_ids = list(graphs.keys())
    cost_matrix = np.full((len(graph_ids), len(graph_ids)), 0, dtype=object)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for i, j in combinations(range(len(graph_ids)), 2):
            futures[(i, j)] = executor.submit(calculate_GED_parallel, graphs[graph_ids[i]], graphs[graph_ids[j]], nt_dict, cache)

        for (i, j), future in futures.items():
            cost_matrix[i, j] = cost_matrix[j, i] = future.result()

    print(f"Total time: {t.time() - basetime}")
    return np.matrix(cost_matrix)